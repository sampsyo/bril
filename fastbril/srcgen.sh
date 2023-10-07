#!/bin/bash

IF=$(echo $1 | awk -F'[/.]' '{printf "config/%s.cf", $2}')
./srcgen.awk $IF > $1
cp $1 lib/
