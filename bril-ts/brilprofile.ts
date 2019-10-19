#!/usr/bin/env node
import * as bril from './bril';
import * as brili from './brili';
import { readStdin } from './util';
import * as fs from 'fs';

async function profile() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  const profile_filename = process.argv[2];
  const profile_data = fs.readFileSync(profile_filename, 'utf8');
  const inputs = profile_data.split('\n');
  for (let input of inputs) {
    const args = input.split(/(\s+)/).filter((e: string) => e.trim().length > 0);
    if (args.length === 0)
      continue;
    let cliArgs = [] as (BigInt | Boolean)[];
    for (let arg of args) {
      if (brili.isNumeric(arg)) {
        cliArgs.push(BigInt(parseInt(arg)));
      } else if (arg === "true" || arg === "false") {
        cliArgs.push(arg === "true");
      } else {
        throw `Argument ${arg} is not of type int or bool; exiting.`;
      }
    }
    brili.evalProg(prog, cliArgs);
  }
  console.log(brili.weighted_call_graph);
}

if (!module.parent) {
  profile();
}

