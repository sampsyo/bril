This is a totally basic test for the text representation.
Here's the program, translated to Bril's JSON representation:

  $ echo 'var x = 5; console.log(x);' | ts2bril > test.json

Here's the text representation:

  $ python $TESTDIR/../bril-txt/bril-json2txt.py < test.json
  main {
    %0 = const 5
    x = id %0
    %1 = id x
    %2 = print %1
  }

Now we ensure that the translation back to JSON still works:

  $ python $TESTDIR/../bril-txt/bril-json2txt.py < test.json > test.bril
  $ python $TESTDIR/../bril-txt/bril-txt2json.py < test.bril | brili
  5
