#!/bin/sh
SVG_VIEWER=firefox
set -e
cd "`dirname $0`"
opam install "."
for f in "$@"
do
  echo "Optimizing example $f.bril"
  bril2json <"$f.bril" >"$f.json"
  if [ -f "$f-before.dot" ]; then rm "$f-before.dot"; fi
  if [ -f "$f-after.dot" ]; then rm "$f-after.dot"; fi
  lcm -dot "$f" "$f.json"
  if [ -f "$f-before.svg" ]; then rm "$f-before.svg"; fi
  dot -Tsvg -o "$f-before.svg" "$f-before.dot"
  if [ -f "$f-after.svg" ]; then rm "$f-after.svg"; fi
  dot -Tsvg -o "$f-after.svg" "$f-after.dot"
  $SVG_VIEWER "$f-before.svg"
  $SVG_VIEWER "$f-after.svg"
done
