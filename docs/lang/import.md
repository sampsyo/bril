Import
======

Core Bril supports bril programs which are self contained, as in they only reference functions defined within that program. The import extension allows for a Bril program to use functions defined in other Bril programs.

A Bril import contains two parts: a file name which can be a relative path specifying the Bril program from which functions can be found and a list of zero or more names of functions. Each function name that is imported can optionally be aliased with a new name.

If a function is imported, it can then be used in the Bril program as if it were defined within the program like any other function. For example, one can import a function and then use it in a `call` instruction.

Well-formed Bril programs that use the import extension always import functions that exist and avoid importing functions that share the same name as another function in scope by giving them a unique alias. If an alias is provided then that function can only be called by the alias.

Implementations of this extension must define a method of resolving the relative paths of the programs being imported to files. This could, but is not required to, be implemented as a list of user-defined library directories or as relative to some source file path. An implementation is only required to support importing the JSON representation of Bril programs.

Bril programs that use the import extension can also be imported and should maintain the above semantics. The programmer is allowed to create cycles of imports.

Syntax
------

The Bril JSON representation is as shown:

    {
        "functions": [
            …
        ],
        "imports": [
            {
                "functions": [
                    {
                        "name": "AND"
                    },
                    {
                        "alias": "LIB_OR",
                        "name": "OR"
                    }
                ],
                "path": "benchmarks/bitwise-ops.bril"
            }
        ]
    }

The grammar of the Bril text representation is then:

    'from' PATH 'import' (FUNC ('as' FUNC)? ',')* ';'

With an example being:

    from "benchmarks/bitwise-ops.bril" import @AND, @OR as @LIB_OR;
