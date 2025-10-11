; JavaScript function extraction queries
; Captures function declarations, expressions, and arrow functions

; Function declarations
(function_declaration
  name: (identifier) @function.name
  body: (statement_block) @function.body) @function.def

; Function expressions
(function_expression
  body: (statement_block) @function.body) @function.def

; Arrow functions (assigned to variables)
(variable_declarator
  name: (identifier) @function.name
  value: (arrow_function) @function.body) @function.def

; Method definitions in classes/objects
(method_definition
  name: (property_identifier) @method.name
  body: (statement_block) @method.body) @method.def

; Import statements for context
(import_statement) @import
