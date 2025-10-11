; Go function extraction queries
; Captures function declarations and method declarations

; Function declarations
(function_declaration
  name: (identifier) @function.name
  body: (block) @function.body) @function.def

; Method declarations
(method_declaration
  name: (field_identifier) @method.name
  body: (block) @method.body) @method.def

; Import statements for context
(import_declaration) @import
