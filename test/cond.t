A simple program with conditionals:

  $ ts2bril < $TESTDIR/cond.ts | brili
  2

Round-trip it through the text format:

  $ ts2bril < $TESTDIR/cond.ts | python $TESTDIR/../bril-txt/bril2txt.py | python $TESTDIR/../bril-txt/txt2bril.py | brili
  2
