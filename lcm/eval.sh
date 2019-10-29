#!/bin/sh
set -e
cd "`dirname $0`"
cd "../test/lcm/"
for f in "$@"
do
  echo "Optimizing example $f.bril"
  bril2json <"$f.bril" >"$f.json"
  lcm "$f.json" >"$f-opt.json"
  brili <"$f-opt.json"
done
