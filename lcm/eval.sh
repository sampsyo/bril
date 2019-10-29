#!/bin/sh
set -e
for f in "$@"
do
  echo
  echo "Optimizing example $f.bril"
  bril2json <"$f.bril" >"$f.json"
  lcm "$f.json" >"$f-opt.json"
  brili <"$f-opt.json"
done
