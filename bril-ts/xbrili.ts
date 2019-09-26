#!/usr/bin/env node
import * as bril from './bril';
import {readStdin, unreachable, StringifyingMap} from './util';
//require('ts-priority-queue');

const argCounts: {[key in bril.OpCode]: number | null} = {
  add: 2,
  mul: 2,
  sub: 2,
  div: 2,
  id: 1,
  lt: 2,
  le: 2,
  gt: 2,
  ge: 2,
  eq: 2,
  not: 1,
  and: 2,
  or: 2,
  print: null,  // Any number of arguments.
  br: 3,
  jmp: 1,
  ret: 0,
  nop: 0,
  // PROB
  flip: 0,
  obv: 1
};


type Value = boolean | BigInt;
type Env = Map<bril.Ident, Value>;
class EnvMap<T> extends StringifyingMap<Env, T> {
  protected stringifyKey(key : Env): string  {
    return JSON.stringify( Array.from(key).sort() );
  }
}

function get(env: Env, ident: bril.Ident) {
  let val = env.get(ident);
  if (typeof val === 'undefined') {
    throw `undefined variable ${ident}`;
  }
  return val;
}

/**
 * Ensure that the instruction has exactly `count` arguments,
 * throwing an exception otherwise.
 */
function checkArgs(instr: bril.Operation, count: number) {
  if (instr.args.length != count) {
    throw `${instr.op} takes ${count} argument(s); got ${instr.args.length}`;
  }
}

function getInt(instr: bril.Operation, env: Env, index: number) {
  let val = get(env, instr.args[index]);
  if (typeof val !== 'bigint') {
    throw `${instr.op} argument ${index} must be a number`;
  }
  return val;
}

function getBool(instr: bril.Operation, env: Env, index: number) {
  let val = get(env, instr.args[index]);
  if (typeof val !== 'boolean') {
    throw `${instr.op} argument ${index} must be a boolean`;
  }
  return val;
}


/**
 * The thing to do after interpreting an instruction: either transfer
 * control to a label, go to the next instruction, or end thefunction.
 */
type PCAction =
  { label : bril.Ident } |
  { next : true } |
  { end : true } |
  { restart : true };
let PC_NEXT: PCAction = {"next": true};
let PC_END: PCAction = {"end": true};
let PC_RESTART: PCAction = {"restart": true};

type SplitAction = 
  { det : true } | { "newenvs" : [Env, number][] };

let ALONE: SplitAction = { det : true}

type Action = PCAction & SplitAction
let NEXT : Action = {...PC_NEXT,...ALONE };
let END : Action = {...PC_END,...ALONE };
let BYE: Action = { newenvs : [], ...PC_RESTART };



/**
 * Interpret an instruction in a given environment, possibly updating the
 * environment. If the instruction branches to a new label, return that label;
 * otherwise, return "next" to indicate that we should proceed to the next
 * instruction or "end" to terminate the function.
 */
function evalInstr(instr: bril.Instruction, env: Env, buffer: any[][]): Action {
  // Check that we have the right number of arguments.
  if (instr.op !== "const") {
    let count = argCounts[instr.op];
    if (count === undefined) {
      throw "unknown opcode " + instr.op;
    } else if (count !== null) {
      checkArgs(instr, count);
    }
  }

  switch (instr.op) {
  case "const":
    // Ensure that JSON ints get represented appropriately.
    let value: Value;
    if (typeof instr.value === "number") {
      value = BigInt(instr.value);
    } else {
      value = instr.value;
    }
    env.set(instr.dest, value);
    return NEXT;

  case "id": {
    let val = get(env, instr.args[0]);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "add": {
    let val = getInt(instr, env, 0) + getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "mul": {
    let val = getInt(instr, env, 0) * getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "sub": {
    let val = getInt(instr, env, 0) - getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "div": {
    let val = getInt(instr, env, 0) / getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "le": {
    let val = getInt(instr, env, 0) <= getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "lt": {
    let val = getInt(instr, env, 0) < getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "gt": {
    let val = getInt(instr, env, 0) > getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "ge": {instr.dest, Math.random() < 0.5
    let val = getInt(instr, env, 0) >= getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "eq": {
    let val = getInt(instr, env, 0) === getInt(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "not": {
    let val = !getBool(instr, env, 0);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "and": {
    let val = getBool(instr, env, 0) && getBool(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "or": {
    let val = getBool(instr, env, 0) || getBool(instr, env, 1);
    env.set(instr.dest, val);
    return NEXT;
  }

  case "print": {
    let values = instr.args.map(i => get(env, i).toString());
    buffer.push(values);
    return NEXT;
  }

  case "jmp": {
    return {"label": instr.args[0], ...ALONE};
  }

  case "br": {
    let cond = getBool(instr, env, 0);
    if (cond) {
      return {"label": instr.args[1], ...ALONE};
    } else {
      return {"label": instr.args[2], ...ALONE};
    }
  }
  
  case "ret": {
    return END;
  }

  case "nop": {
    return NEXT;
  }
  
  case "flip": {
    let newE1 = new Map(env); // clone env, do both.
    let newE2 = new Map(env); // clone env, do both.
    newE1.set(instr.dest, true);
    newE2.set(instr.dest, false); 
    return { newenvs : [[newE1, 0.5], [newE2, 0.5]], ...NEXT};
  }
  
  case "obv": {
    let cond = getBool(instr, env, 0);
    return cond ? NEXT : BYE;
  }
  }
  unreachable(instr);
  throw `unhandled opcode ${(instr as any).op}`;
}

type ProgPt = [number, Env]; // program counter, environment
class Path extends StringifyingMap<ProgPt, [ProgPt | undefined, number]> {
  protected stringifyKey(key : ProgPt): string { return pt2str(key); }
}
// singlely linked list going backwards in time
type PointedPath = { head : [ProgPt, number], hist : Path }
// PointedPath hist includes the head.

function pt2str(pt : ProgPt) : string {
  return JSON.stringify( [pt[0], Array.from(pt[1]).sort()]);
}
/**
* Extend the linked list with a point
*/
function extend(path : PointedPath, pt : ProgPt, p : number) { 
  let prev_head = path.head;
  path.hist.set( pt, prev_head )
  path.head = [pt, p * prev_head[1] /* multiply probability */ ];
}
function path_copy(path: PointedPath ) {
  return { head : path.head, hist : new Path(path.hist) };
}

function evalFunc(func: bril.Function, buffer: any[][] ) {
  let cur_pt : ProgPt = [0, new Map()];
  let paths : PointedPath[] = [
    { head : [cur_pt, 1], hist : new Path([[cur_pt, [undefined, 1]]]) } ];
  
  let probs : Map<ProgPt, number> = new Map();
  // also, shift.  
  let finished : Map<Env, number> = new Map();
  let missing_prob = 0;
  //let transition : Map< ProgPt, Map<ProgPt, number>> = new Map();

  let n_instrs_at_once = 1;
  let pathidx = 0;
  
  while(paths.length > pathidx) {
    // find run with highest mass     
    let [[i, penv], pr] = paths[pathidx].head;
    let env = new Map(penv);
    let cpr = 1; // conditional probability of next instruction given current one.
    //curmass = new Map([...curmass].shift()); // very slow, do something else eventually
    
    //for (let i = 0; i < func.instrs.length; ++i) {
    let line = func.instrs[i];
    
    if ('op' in line) {
      let action = evalInstr(line, env, buffer);
      
      // handle motion form PCActions
      if ('label' in action) {
        // Search for the label and transfer control.
        for (i = 0; i < func.instrs.length; ++i) {
          let sLine = func.instrs[i];
          if ('label' in sLine && sLine.label === action.label) {
            break;
          }
        }
        
        
        if (i === func.instrs.length) {
          throw `label ${action.label} not found`;
        }
      } else if ('end' in action) {
        pathidx ++; 
        finished.set(env, (finished.get(env) || 0) + pr)
        continue;
      } else if ('next' in action) {
        i++;
        if (i == func.instrs.length) { 
          pathidx ++;
          finished.set(env, (finished.get(env) || 0) + pr)
          continue; 
        }
      }
      
      //handle world splitting and env copying
      if ( 'newenvs' in action ) {
        let totalp = 0;
        for ( let [e, p] of action.newenvs) {
          totalp += p;
          let copy = path_copy(paths[pathidx]);
          extend(copy, [i, e], p);
          paths.push(copy);
        }
        
        if(totalp > 1){
          console.warn("No convergence guarantees if positive weighting can occur.")
        } else {
          missing_prob += pr * (1-totalp);
        }
        
        pathidx ++;
        continue;
      }
    } else { // this is a label. Copy environment.
      env = new Map(env);
      i++;
    }
    
    if ( paths[pathidx].hist.has([i,env]) ) {
      let factor = 1 / (1 - pr / paths[pathidx].hist.get([i,env])![1]);
      console.log('head: ', paths[pathidx].head[0], '[i,env]: ', [i,env],  paths[pathidx].hist.has(paths[pathidx].head[0]));
      let [pt, pro] = paths[pathidx].hist.get(paths[pathidx].head[0])!;
      
      while( pt !== undefined && pt2str(pt) != pt2str([i,env]) ){
        [pt, pro] = paths[pathidx].hist.get(pt)!;
      }
      
      pathidx ++;
      continue;
      //console.log(`shows up in paths[${pathidx}]`, paths[pathidx].hist)
    } else {
      extend(paths[pathidx], [i,env], cpr);
    }
    
    //}
  }
  finished.forEach( (p,e) => finished.set(e, p / (1-missing_prob)))
  
  console.log('paths', paths);
  console.log('finished');
  finished.forEach( (p, e) =>  console.log(e, 'prob = ', p) );
}
function evalProg(prog: bril.Program) {
  let buffer : any[][] = [];
  
  for (let func of prog.functions) {
    if (func.name === "main") {
      evalFunc(func, buffer);
    }
  }
  
  // print buffer
  for(let i = 0; i < buffer.length; i++) {
    console.log(...buffer[i]);
  }
}

async function main() {
  let prog = JSON.parse(await readStdin()) as bril.Program;
  evalProg(prog);
}

// Make unhandled promise rejections terminate.
process.on('unhandledRejection', e => { throw e });

main();
