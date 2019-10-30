#!/bin/sh
set -e
echo -n "program," >&2
echo -n "ops (before lcm)," >&2
echo -n "computations (before lcm)," >&2
echo -n "ops (after lcm)," >&2
echo -n "computations (after lcm)\n" >&2
for f in "$@"
do
  bril2json <"$f.bril" >"$f.json"
  lcm "$f.json" >"$f-opt.json"
  echo -n "$f," >&2
  brili <"$f.json"
  echo -n "," >&2
  brili <"$f-opt.json"
  echo >&2
done
