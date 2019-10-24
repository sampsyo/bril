#!/usr/bin/env bash
prog=""
if [ $# -eq 1 ]; then
  echo "Interpreting with no optimization"
  prog=$(cat "$1" | bril2json)
elif [ $# -eq 2 ]; then
  echo "Interpreting with optimization"
  prog=$(cat "$1" | bril2json | python ../tce.py)
else
  echo "Wrong arguments. Usage: ./test.sh prog.bril [tce]"
fi
/usr/bin/time -l sh -c 'cat prog | brili'
