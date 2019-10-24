#!/usr/bin/env bash
if [ $# -eq 1 ]; then
  echo "Interpreting with no optimization"
  cat "$1" | bril2json | brili
elif [ $# -eq 2 ]; then
  echo "Interpreting with optimization"
  cat "$1" | bril2json | python ../tce.py | brili
else
  echo "Wrong arguments. Usage: ./test.sh prog.bril [tce]"
fi
