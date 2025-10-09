# MIT License
#
# Copyright (c) 2024 Semantic Code Analyzer Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Contrastive learning trainer for fine-tuning RobertaModel.

This module implements contrastive learning to fine-tune RobertaModel
(NOT RobertaForMaskedLM) on custom codebases. Uses the same model class
that works for inference on MPS.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

from ..hardware import DeviceManager, DeviceType
from .data_preparation import CodeDatasetPreparator
from .pair_generation import CodePair, ContrastivePairGenerator

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning GraphCodeBERT with contrastive learning."""

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Contrastive learning parameters
    temperature: float = 0.07  # Temperature for InfoNCE loss
    pairs_per_sample: int = 3  # Positive/negative pairs per code sample

    # Data parameters
    max_files: int = 1000
    validation_split: float = 0.1
    include_test_files: bool = False
    include_generated_files: bool = False

    # Hardware parameters
    device_preference: str = "auto"  # auto, cpu, mps, cuda
    gradient_accumulation_steps: int = 4

    # Model parameters
    model_name: str = "microsoft/graphcodebert-base"
    model_revision: str = "main"  # pragma: allowlist secret

    # Output parameters
    save_steps: int = 500
    logging_steps: int = 100


class ContrastiveTripletDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset for contrastive learning with triplets.

    Each sample contains an anchor, positive (similar), and negative (dissimilar) code.
    """

    def __init__(
        self,
        pairs: list[CodePair],
        tokenizer: RobertaTokenizer,
        max_length: int = 512,
    ):
        """Initialize contrastive dataset.

        Args:
            pairs: List of code pairs
            tokenizer: RobertaTokenizer
            max_length: Maximum sequence length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Separate positive and negative pairs
        self.positive_pairs = [p for p in pairs if p.is_positive]
        self.negative_pairs = [p for p in pairs if not p.is_positive]

        logger.info(
            f"Created dataset with {len(self.positive_pairs)} positive and "
            f"{len(self.negative_pairs)} negative pairs"
        )

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
            Number of samples
        """
        return len(self.positive_pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a triplet (anchor, positive, negative).

        Args:
            idx: Sample index

        Returns:
            Dictionary with tokenized anchor, positive, and negative
        """
        positive_pair = self.positive_pairs[idx]

        # Get a random negative pair for this anchor
        negative_pair = random.choice(self.negative_pairs)  # nosec B311

        # Tokenize all three
        anchor_enc = self.tokenizer(
            positive_pair.anchor.content,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        positive_enc = self.tokenizer(
            positive_pair.comparison.content,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        negative_enc = self.tokenizer(
            negative_pair.comparison.content,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor_enc["input_ids"].squeeze(),
            "anchor_attention_mask": anchor_enc["attention_mask"].squeeze(),
            "positive_input_ids": positive_enc["input_ids"].squeeze(),
            "positive_attention_mask": positive_enc["attention_mask"].squeeze(),
            "negative_input_ids": negative_enc["input_ids"].squeeze(),
            "negative_attention_mask": negative_enc["attention_mask"].squeeze(),
        }


class CodeStyleTrainer:
    """Contrastive trainer for fine-tuning RobertaModel (base encoder only).

    This trainer uses RobertaModel instead of RobertaForMaskedLM to ensure
    MPS compatibility and directly optimizes embeddings for similarity measurement.
    """

    def __init__(
        self,
        config: FineTuningConfig,
        repo_path: str,
        cache_dir: Path | None = None,
        device_manager: DeviceManager | None = None,
    ):
        """Initialize the contrastive trainer.

        Args:
            config: Fine-tuning configuration
            repo_path: Path to git repository
            cache_dir: Directory for caching models
            device_manager: Device manager for hardware acceleration
        """
        self.config = config
        self.repo_path = Path(repo_path)
        self.cache_dir = cache_dir or Path.cwd() / ".sca_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize device manager
        if device_manager is None:
            device_pref = (
                None
                if config.device_preference == "auto"
                else DeviceType(config.device_preference.lower())
            )
            self.device_manager = DeviceManager(prefer_device=device_pref)
        else:
            self.device_manager = device_manager

        self.device = self.device_manager.torch_device
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {config.model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(
            config.model_name,
            revision=config.model_revision,
            cache_dir=str(self.cache_dir),
        )  # nosec B615

        # Model will be loaded during training
        self.model: RobertaModel | None = None

        # Training state
        self.training_stats: dict[str, Any] = {}

    def fine_tune_on_commit(
        self, commit_hash: str, output_name: str | None = None
    ) -> Path:
        """Fine-tune RobertaModel using contrastive learning on a commit.

        Args:
            commit_hash: Git commit hash to train on
            output_name: Optional custom name for output

        Returns:
            Path to saved fine-tuned model
        """
        start_time = time.time()
        logger.info(f"Starting contrastive fine-tuning on commit {commit_hash}")

        # Step 1: Extract code
        logger.info("Step 1/5: Extracting code from commit...")

        preparator = CodeDatasetPreparator(
            repo_path=str(self.repo_path),
            tokenizer=self.tokenizer,
            max_files=self.config.max_files,
            include_test_files=self.config.include_test_files,
            include_generated_files=self.config.include_generated_files,
        )

        code_samples = preparator.extract_code_from_commit(commit_hash)

        if not code_samples:
            raise ValueError(f"No code samples found in commit {commit_hash}")

        logger.info(f"Extracted {len(code_samples)} code samples")

        # Step 2: Generate pairs
        logger.info("Step 2/5: Generating contrastive pairs...")

        pair_generator = ContrastivePairGenerator()
        pairs = pair_generator.generate_pairs(
            code_samples, pairs_per_sample=self.config.pairs_per_sample
        )

        logger.info(f"Generated {len(pairs)} training pairs")

        # Create dataset
        dataset = ContrastiveTripletDataset(
            pairs=pairs, tokenizer=self.tokenizer, max_length=512
        )

        # Split into train/val
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        from torch.utils.data import random_split

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        logger.info(f"Split: {train_size} train, {val_size} val")

        # Step 3: Load RobertaModel (NOT RobertaForMaskedLM - works on MPS!)
        logger.info("Step 3/5: Loading RobertaModel (base encoder)...")

        self.model = RobertaModel.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            cache_dir=str(self.cache_dir),
            low_cpu_mem_usage=True,
        )  # nosec B615

        self.model.to(self.device)
        self.model.train()

        logger.info(f"Model loaded and moved to {self.device}")

        # Step 4: Train with contrastive loss
        logger.info("Step 4/5: Training model...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )

        self.training_stats = self._train_contrastive(train_loader, val_loader)

        # Step 5: Save
        logger.info("Step 5/5: Saving fine-tuned model...")

        output_dir = self._save_model(commit_hash, output_name)

        training_time = time.time() - start_time
        logger.info(
            f"Contrastive fine-tuning completed in {training_time / 60:.1f} minutes"
        )

        return output_dir

    def _train_contrastive(
        self, train_loader: DataLoader[Any], val_loader: DataLoader[Any]
    ) -> dict[str, Any]:
        """Train with contrastive loss.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader

        Returns:
            Training statistics
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        # Scheduler
        total_steps = len(train_loader) * self.config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

        stats: dict[str, Any] = {
            "train_losses": [],
            "val_losses": [],
            "epochs": self.config.epochs,
        }

        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")

            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, epoch)
            stats["train_losses"].append(train_loss)

            # Validate
            val_loss = self._validate_epoch(val_loader)
            stats["val_losses"].append(val_loss)

            logger.info(
                f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Log MPS status after each epoch if on MPS device
            if self.device.type == "mps":
                try:
                    logger.info(
                        f"Epoch {epoch + 1} - MPS device active, training completed successfully"
                    )
                except Exception as e:
                    logger.debug(f"Could not log MPS status: {e}")

        return stats

    def _train_epoch(
        self,
        train_loader: DataLoader[Any],
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
    ) -> float:
        """Train one epoch with contrastive loss.

        Args:
            train_loader: Training dataloader
            optimizer: Optimizer
            scheduler: LR scheduler
            epoch: Current epoch

        Returns:
            Average training loss
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            # Move to device
            anchor_ids = batch["anchor_input_ids"].to(self.device)
            anchor_mask = batch["anchor_attention_mask"].to(self.device)
            pos_ids = batch["positive_input_ids"].to(self.device)
            pos_mask = batch["positive_attention_mask"].to(self.device)
            neg_ids = batch["negative_input_ids"].to(self.device)
            neg_mask = batch["negative_attention_mask"].to(self.device)

            # Get embeddings (CLS token)
            anchor_emb = self.model(
                input_ids=anchor_ids, attention_mask=anchor_mask
            ).last_hidden_state[:, 0, :]
            pos_emb = self.model(
                input_ids=pos_ids, attention_mask=pos_mask
            ).last_hidden_state[:, 0, :]
            neg_emb = self.model(
                input_ids=neg_ids, attention_mask=neg_mask
            ).last_hidden_state[:, 0, :]

            # Contrastive loss (InfoNCE)
            loss = self._contrastive_loss(anchor_emb, pos_emb, neg_emb)

            # Gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            progress_bar.set_postfix({"loss": loss.item()})

        # Synchronize MPS if applicable
        if self.device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception as e:
                logger.debug(f"MPS synchronize failed: {e}")

        return total_loss / num_batches

    def _validate_epoch(self, val_loader: DataLoader[Any]) -> float:
        """Validate with contrastive loss.

        Args:
            val_loader: Validation dataloader

        Returns:
            Average validation loss
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                anchor_ids = batch["anchor_input_ids"].to(self.device)
                anchor_mask = batch["anchor_attention_mask"].to(self.device)
                pos_ids = batch["positive_input_ids"].to(self.device)
                pos_mask = batch["positive_attention_mask"].to(self.device)
                neg_ids = batch["negative_input_ids"].to(self.device)
                neg_mask = batch["negative_attention_mask"].to(self.device)

                anchor_emb = self.model(
                    input_ids=anchor_ids, attention_mask=anchor_mask
                ).last_hidden_state[:, 0, :]
                pos_emb = self.model(
                    input_ids=pos_ids, attention_mask=pos_mask
                ).last_hidden_state[:, 0, :]
                neg_emb = self.model(
                    input_ids=neg_ids, attention_mask=neg_mask
                ).last_hidden_state[:, 0, :]

                loss = self._contrastive_loss(anchor_emb, pos_emb, neg_emb)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]

        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Compute similarities
        pos_sim = F.cosine_similarity(anchor, positive, dim=1) / self.config.temperature
        neg_sim = F.cosine_similarity(anchor, negative, dim=1) / self.config.temperature

        # InfoNCE loss: -log(exp(pos) / (exp(pos) + exp(neg)))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=self.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def _save_model(self, commit_hash: str, output_name: str | None = None) -> Path:
        """Save fine-tuned RobertaModel.

        Args:
            commit_hash: Source commit hash
            output_name: Optional output name

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        model_name = output_name or commit_hash[:7]
        output_dir = self.cache_dir / "fine_tuned_models" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving fine-tuned RobertaModel to {output_dir}")

        # Save RobertaModel directly (same class used for inference!)
        self.model.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)

        # Save metadata
        metadata = {
            "commit_hash": commit_hash,
            "model_name": self.config.model_name,
            "training_method": "contrastive_learning",
            "training_config": asdict(self.config),
            "training_stats": self.training_stats,
            "timestamp": time.time(),
            "device": str(self.device),
        }

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return output_dir
