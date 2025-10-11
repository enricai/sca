; Rust function extraction queries
; Captures function items and impl methods

; Function items
(function_item
  name: (identifier) @function.name
  body: (block) @function.body) @function.def

; Methods in impl blocks
(impl_item
  body: (declaration_list
    (function_item
      name: (identifier) @method.name
      body: (block) @method.body) @method.def))

; Use declarations for context
(use_declaration) @import
