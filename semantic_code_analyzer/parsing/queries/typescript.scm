; TypeScript function extraction queries
; Extends JavaScript queries with TypeScript-specific syntax

; Function declarations
(function_declaration
  name: (identifier) @function.name
  body: (statement_block) @function.body) @function.def

; Function expressions
(function_expression
  body: (statement_block) @function.body) @function.def

; Arrow functions
(variable_declarator
  name: (identifier) @function.name
  value: (arrow_function) @function.body) @function.def

; Method definitions
(method_definition
  name: (property_identifier) @method.name
  body: (statement_block) @method.body) @method.def

; TypeScript-specific: Interface method signatures (no body)
; We skip these as they don't have implementations

; Import statements for context
(import_statement) @import
