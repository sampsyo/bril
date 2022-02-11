#!/bin/bash

rm -r "$1-gold";
mkdir "$1-gold";

QUOTE='"';
echo "command = $QUOTE bril2json < {filename} | python3 ../../$1.py | brili {args} $QUOTE" >> "$1-gold"/turnt.toml;

for prog in $1/*.bril;
 do
    outname=${prog%.*};
    outname=${outname#*/};
    cat $prog | grep -v 'CMD' | grep -v 'ARGS' > "$1-gold/$outname.bril";
    bril2json < $prog | brili > "$1-gold/$outname.out" 2> /dev/null;
    bril2json < $prog | brili -p 2> "$1-gold/$outname.prof" > /dev/null;
done
