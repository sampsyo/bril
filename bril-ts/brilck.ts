#!/usr/bin/env node
import * as bril from './bril';
import {readStdin} from './util';

function checkProg(prog: bril.Program) {
  console.log(prog);
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  checkProg(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
