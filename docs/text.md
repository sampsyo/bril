Bril Text Format
================

While Bril's canonical representation is a JSON AST, humans don't like to read and write JSON.
To accommodate our human foibles, we also have a simple textual representation.
There is a parser and pretty printer tool that can convert the text representation to and from JSON.

For example, this Bril program in JSON:

    {
      "functions": [{
        "name": "main",
        "instrs": [
          { "op": "const", "type": "int", "dest": "v0", "value": 1 },
          { "op": "const", "type": "int", "dest": "v1", "value": 2 },
          { "op": "add", "type": "int", "dest": "v2", "args": ["v0", "v1"] },
          { "op": "print", "args": ["v2"] }
        ]
      }]
    }

Gets represented in text like this:

    main {
      v0: int = const 1;
      v1: int = const 2;
      v2: int = add v0 v1;
      print v2;
    }
