#!/bin/sh
SVG_VIEWER=firefox
set -e
cd "`dirname $0`"
opam install "."
cd "../test/lcm/"
for f in "$@"
do
  echo "Optimizing example $f.bril"
  bril2json <"$f.bril" >"$f.json"
  lcm -dot "$f" "$f.json"
  dot -Tsvg -o "$f-before.svg" "$f-before.dot"
  dot -Tsvg -o "$f-after.svg" "$f-after.dot"
  $SVG_VIEWER "$f-before.svg"
  $SVG_VIEWER "$f-after.svg"
done
