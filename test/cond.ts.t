A simple program with conditionals:

  $ ts2bril < $TESTDIR/cond.ts | brili
  2

Round-trip it through the text format:

  $ ts2bril < $TESTDIR/cond.ts | python $TESTDIR/../bril-txt/bril-json2txt.py | python $TESTDIR/../bril-txt/bril-txt2json.py | brili
  2
