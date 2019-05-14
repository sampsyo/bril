Bril in TypeScript
==================

This directory contains some tools written in TypeScript for dealing with Bril.
To use them, you will need [Node][] and [Yarn][].
(On macOS, type `brew install node yarn`.)
I recommend you do this:

    $ yarn
    $ yarn build
    $ yarn link

The last thing will install symlinks to the two utility programs---for me, they ended up in `/usr/local/bin`.
The tools are `ts2bril`, which compiles TypeScript to Bril, and `brili`, an interpreter.
Both of them expect input on stdin and send results to stdout.

There is a definition of the Bril language as a TypeScript data type declaration, in `bril.ts`.

[node]: https://nodejs.org/en/
[yarn]: https://yarnpkg.com/en/
