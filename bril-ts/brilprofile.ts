#!/usr/bin/env node
import * as bril from './bril';
import * as brili from './brili';
import * as callgraph from './call-graph';
import { readStdin } from './util';
import * as fs from 'fs';

function json_from_call_graph(call_graph: callgraph.WeightedCallGraph) {
  let call_graph_json: { from: string; to: string; count: Number; }[] = [];
  call_graph.forEach((value: Number, key: string) => {
    const vertices = callgraph.getVerticesFromEdge(key);
    call_graph_json.push({
      from: vertices[0],
      to: vertices[1],
      count: value
    });
  });
  return call_graph_json;
}

async function profile() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  const profile_filename = process.argv[2];
  const profile_data = fs.readFileSync(profile_filename, 'utf8');
  const inputs = profile_data.split('\n');
  let consoleLog = console.log;
  console.log = function () { };
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
  console.log = consoleLog;
  let call_graph_json: { from: string; to: string; count: Number; }[] = json_from_call_graph(brili.weighted_call_graph);
  let basic_block_json: { function: string; from: string; to: string; count: Number; }[][] = [];
  brili.basic_block_flows.forEach((value: callgraph.WeightedCallGraph, key: string) => {
    let bb_flows = json_from_call_graph(value);
    basic_block_json.push(bb_flows.map(el => ({...el, function: key})));
  });
  console.log(JSON.stringify({
    call_graph: call_graph_json,
    basic_block_flows: basic_block_json
  }, undefined, 2));
}

if (!module.parent) {
  profile();
}

