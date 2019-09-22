// The compilation procedure is as follows
// Add this to bril-ts/package.json to install nbind {
/*  
{
  "scripts": {
    "autogypi": "autogypi",
    "node-gyp": "node-gyp",
    "emcc-path": "emcc-path",
    "copyasm": "copyasm",
    "ndts": "ndts"
  }
}
*/
// In bril-ts/ install nbind (one time setup for system)
/*
npm install --save \
  nbind autogypi node-gyp
*/
// Then register this file (one time setup for each file)
/*
npm run -- autogypi \
  --init-gyp \
  -p nbind -s ../csrc/hello/hello.cc
*/
// Then create a callable binary (need to recompile everytime)
/*
npm run -- node-gyp \
  configure build
*/
// In your typescript program you can use with
/*
let nbind = require('nbind');
let lib = nbind.init().lib;
 
lib.Greeter.sayHello('you');
*/
// turnt doesn't work because it goes into the test directory and nbind can't find the c binary
// instead forced to use 'bril2json < ../test/proj1/ldst.bril | brili' from bril-ts/ directory 


// normal code compiled by gcc

#include <string>
#include <iostream>

struct Greeter {
  static void sayHello(
    std::string name
  ) {
    std::cout
      << "Hello, "
      << name << "!\n";
  }
};

// binding code required by nbind
// we compile through npm so don't worry about
// a normal gcc compile not knowing where nbind.h is

#include "nbind/nbind.h"

NBIND_CLASS(Greeter) {
    method(sayHello);
}
