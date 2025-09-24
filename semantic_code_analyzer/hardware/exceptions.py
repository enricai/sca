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

"""Custom exceptions for hardware acceleration and device management.

This module provides specific exception types for hardware-related errors
to enable more precise error handling and better user experience.
"""

from __future__ import annotations


class HardwareError(Exception):
    """Base exception for hardware-related errors with actionable suggestions."""

    def __init__(
        self,
        message: str,
        device_type: str | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        """Initialize hardware error with context and suggestions.

        Args:
            message: Error message describing the issue
            device_type: Device type that caused the error (optional)
            suggestions: List of actionable suggestions for the user (optional)
        """
        super().__init__(message)
        self.device_type = device_type
        self.suggestions = suggestions or []

    def get_user_friendly_message(self) -> str:
        """Get user-friendly error message with suggestions.

        Returns:
            Formatted error message with actionable suggestions.
        """
        msg = str(self)
        if self.suggestions:
            msg += "\n\nSuggestions:"
            for suggestion in self.suggestions:
                msg += f"\n  • {suggestion}"
        return msg


class DeviceDetectionError(HardwareError):
    """Exception raised when hardware detection fails."""

    def __init__(self, message: str, detection_method: str | None = None) -> None:
        """Initialize device detection error.

        Args:
            message: Error message describing the detection failure
            detection_method: Method used for detection that failed (optional)
        """
        super().__init__(message)
        self.detection_method = detection_method


class ModelLoadingError(HardwareError):
    """Exception raised when model loading fails on specific hardware."""

    def __init__(
        self,
        message: str,
        device_type: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize model loading error.

        Args:
            message: Error message describing the loading failure
            device_type: Device type that failed to load the model (optional)
            model_name: Name of the model that failed to load (optional)
        """
        super().__init__(message, device_type)
        self.model_name = model_name


class AccelerationUnavailableError(HardwareError):
    """Exception raised when hardware acceleration is not available."""

    def __init__(self, message: str, requested_device: str | None = None) -> None:
        """Initialize acceleration unavailable error.

        Args:
            message: Error message describing why acceleration is unavailable
            requested_device: Requested device type that is unavailable (optional)
        """
        # Add device-specific suggestions
        suggestions = []
        if requested_device and "mps" in requested_device.lower():
            suggestions = [
                "Update PyTorch: pip install --upgrade torch torchvision",
                "Check macOS version compatibility (requires macOS 12.3+)",
                "Use --device cpu flag to disable MPS and run on CPU",
                "Restart terminal session after PyTorch update",
            ]
        elif requested_device and "cuda" in requested_device.lower():
            suggestions = [
                "Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                "Check NVIDIA driver installation",
                "Use --device cpu flag to disable CUDA and run on CPU",
            ]
        else:
            suggestions = [
                "Check hardware acceleration library installation",
                "Use --device cpu flag to run analysis on CPU",
                "Restart terminal session after library updates",
            ]

        super().__init__(message, requested_device, suggestions)
        self.requested_device = requested_device


class FallbackError(HardwareError):
    """Exception raised when fallback mechanisms fail."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
        fallback_device: str | None = None,
    ) -> None:
        """Initialize fallback error.

        Args:
            message: Error message describing the fallback failure
            original_error: Original error that triggered the fallback (optional)
            fallback_device: Device type used for fallback (optional)
        """
        super().__init__(message)
        self.original_error = original_error
        self.fallback_device = fallback_device
