Test the meaningful use of constants in the text representation:

  $ echo 'console.log(1 + 2)' | ts2bril > test.json

The text representation is simple:

  $ python $TESTDIR/../bril-txt/bril2txt.py < test.json
  main {
    %0 = const 1
    %1 = const 2
    %2 = add %0 %1
    %3 = print %2
  }

Check the output:

  $ python $TESTDIR/../bril-txt/bril2txt.py < test.json > test.bril
  $ python $TESTDIR/../bril-txt/txt2bril.py < test.bril > test2.json
  $ cat test2.json
  {
    "functions": [
      {
        "name": "main",
        "instrs": [
          {
            "op": "const",
            "dest": "%0",
            "value": 1
          },
          {
            "op": "const",
            "dest": "%1",
            "value": 2
          },
          {
            "op": "add",
            "dest": "%2",
            "args": [
              "%0",
              "%1"
            ]
          },
          {
            "op": "print",
            "dest": "%3",
            "args": [
              "%2"
            ]
          }
        ]
      }
    ]
  }

Try actually running it:

  $ brili < test2.json
  3
