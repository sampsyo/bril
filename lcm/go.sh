#!/bin/sh
set -e
cd "`dirname $0`"
f="$1"
opam install "."
cd "../test/lcm/"
bril2json <"$f.bril" >"$f.json"
lcm -dot "$f" "$f.json"
dot -Tsvg -o "$f-before.svg" "$f-before.dot"
dot -Tsvg -o "$f-after.svg" "$f-after.dot"
open "$f-before.svg"
open "$f-after.svg"
