#!/bin/bash


CORE=$(echo $1 | awk -F'[/.]' '{print $2}')
IF=$(echo $1 | awk -F'[/.]' '{printf "config/%s.cf", $2}')
if [[ $CORE == "types" ]]; then
    CORE="Types"
else
    CORE="$CORE Instructions"
fi
awk -v header="$CORE" -f docgen.awk $IF > $1
