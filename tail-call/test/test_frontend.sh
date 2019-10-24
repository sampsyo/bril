#!/usr/bin/env bash
if [ $# -eq 1 ]; then
  echo "Interpreting with no optimization"
  ts2bril "$1" | brili
elif [ $# -eq 2 ]; then
  echo "Interpreting with optimization"
  ts2bril "$1" | python ../tce.py | brili
else
  echo "Wrong arguments. Usage: ./test.sh prog.bril [tce]"
fi
