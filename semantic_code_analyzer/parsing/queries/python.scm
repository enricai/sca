; Python function and method extraction queries
; Captures function definitions and class methods

; Class methods (capture first to prioritize)
(class_definition
  body: (block
    (function_definition) @method.def))

; Top-level functions (not inside a class)
(module
  (function_definition) @function.def)

; Import statements for context
(import_statement) @import
(import_from_statement) @import
