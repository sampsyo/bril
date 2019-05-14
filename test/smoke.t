Tiny smoke test for compilation from TypeScript:

  $ echo 'let x = 42' | ts2bril
  {
    "functions": [
      {
        "name": "main",
        "instrs": [
          {
            "op": "const",
            "value": 42,
            "dest": "%0"
          },
          {
            "op": "id",
            "args": [
              "%0"
            ],
            "dest": "x"
          }
        ]
      }
    ]
  } (no-eol)

And the interpreter:

  $ echo '{"functions": [{"name": "main", "instrs": [
  > {"op": "const", "value": 5, "dest": "v"},
  > {"op": "print", "args": ["v"]}]}]}' | brili
  5
