#!/usr/bin/env bash
prog=""
if [ $# -eq 1 ]; then
  echo "Interpreting with no optimization"
  prog=$(ts2bril "$1")
elif [ $# -eq 2 ]; then
  echo "Interpreting with optimization"
  prog=$(ts2bril "$1" | python ../tce.py)
else
  echo "Wrong arguments. Usage: ./test.sh prog.bril [tce]"
  exit
fi
/usr/bin/time -l brili < "$prog"
