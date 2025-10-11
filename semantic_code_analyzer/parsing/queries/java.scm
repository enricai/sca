; Java function extraction queries
; Captures method declarations in classes

; Method declarations
(method_declaration
  name: (identifier) @method.name
  body: (block) @method.body) @method.def

; Import declarations for context
(import_declaration) @import
