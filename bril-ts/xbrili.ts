#!/usr/bin/env node
import * as bril from './bril';
import {readStdin, unreachable, StringifyingMap, Env, Value, env2str} from './util';

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

type Loc = number | "done";
type ProgPt = [Loc, Env]; // program counter, environment
class PtMap<V> extends StringifyingMap<ProgPt, V> {
  protected stringifyKey(key : ProgPt): string { return pt2str(key); }
}

function pt2str(pt : ProgPt) : string {
  return pt[0].toString() + ';'+ env2str(pt[1]);
}


function makeTransFn(func: bril.Function, iobuf : any[][]) {
  function transition(pt : ProgPt) : PtMap<number> {
    let [i, old_env] = pt;
    if( i == "done" || i >= func.instrs.length ) {
      return new PtMap([[[i,old_env], 1]]);
    }

    let line = func.instrs[i];
    let env = new Map(old_env);

    if ('op' in line) {
      let action = evalInstr(line, env, iobuf);

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
        i = "done";
      } else if ('next' in action) {
        i++;
        if (i == func.instrs.length) {
          // pathidx ++;
          // finished.set(env, (finished.get(env) || 0) + pr)
          i = "done";
        }
      }

      //handle world splitting and env copying
      if ( 'newenvs' in action ) {
        let totalp = 0;
        let newDist : PtMap<number> = new PtMap();

        for ( let [e, p] of action.newenvs) {
          totalp += p;
          newDist.set([i,e], p);
        }

        if(totalp > 1){
          console.warn("No convergence guarantees if positive weighting can occur.")
        } else {
          // TODO: missing probability should be updated globally.
          //missing_prob += pr * (1-totalp);
        }

        return newDist;
      }

    } else { // this is a label. 
      i++;
    }

    // TODO: optimize this by reusing map.
    // now return the distribution over next instructions
    return new PtMap([[[i,env], 1]]);
  }
  return transition;
}


function evalFunc(func: bril.Function, buffer: any[][],
      maxQLen: number = 0,  tol : number = 0)
{
  let best : PtMap<PtMap<number>> = new PtMap();
  let finished = new Set<string>(); // stringifiied points.
  let probs : PtMap<number> = new PtMap();
  let missing_prob = 0;
  
  function nodenumber(n : ProgPt) {
    return probs.keyList().findIndex(pt => pt2str(pt) == pt2str(n));
  }

  // TODO: make this a priority queue
  type Task = "explore" | "merge";
  let NEW : Task = "explore";
  let START : ProgPt = [0, new Map()];

  let queue : PtMap<Task> = new PtMap([[START, NEW]]);
  probs.set(START, 1);
  let transition = makeTransFn(func,buffer);

  while(queue.size() > 0) {
    // console.log(queue.size(), queue.keys());
    // console.log('\nQueue Size: ', queue.size());
    // queue.keyList().forEach( q =>  console.log('  pt ', nodenumber(q), ' \t ', q, probs.get(q), queue.get(q))  );

    // TODO: potential optimization: find run with highest mass?
    let [current_pt, task] = queue.pop_first()!;
    
    if( finished.has(pt2str(current_pt)) ||
      probs.getOr(current_pt,1) < tol && tol > 0) {
      continue;
    }


    let dist = best.get(current_pt) || transition(current_pt);

    let twohop : PtMap<number> = new PtMap();

    // do monad multiplication!
    for ( let p of dist.keys() ) {
      let pdist = best.get(p) || transition(p);
      for ( let q of pdist.keys() ) {
        twohop.set(q, (twohop.get(q) || 0) + dist.get(p)! * pdist.get(q)! );
      }
    }

    // update probs
    for ( let p of twohop.keys() ) {
        probs.set(p, Math.min(probs.getOr(p, 1), twohop.get(p)!));
    }

    // limit computation!
    if( twohop.has(current_pt) ) {
      let factor = 1 / (1 - twohop.get(current_pt)!);
      for ( let pt of twohop.keys() ) {
        if ( pt2str(pt) != pt2str(current_pt) ) {
          twohop.set(pt, twohop.get(pt)! * factor)
        }
      }

      if(twohop.get(current_pt)! < 1) {
        twohop.delete(current_pt);
      }
    }

    // book-keeping: update best.
    let is_same = best.has(current_pt) &&
        ( twohop.size() == best.get(current_pt)!.size() );
    if( is_same ) {
      let best_here = best.get(current_pt)!;
      for (let p of twohop.keys()) {
        if( ! ( best_here.has(p) && best_here.get(p) == twohop.get(p) ) ) {
          is_same = false;
          break;
        }
      }
    }

    if (is_same) {
      finished.add(pt2str(current_pt));
    }
    else {
      if ( (!maxQLen || (queue.size() < maxQLen-1)) &&
            ( tol <= 0 || (probs.getOr(current_pt, 1) > tol ))
      ) {
        best.set(current_pt, twohop);
        let children_enqueued = false;
        
        for (let k of twohop.keys() ) {
          let estp = Math.min(probs.getOr(k, 1) ,
                        probs.getOr(current_pt, 1) * twohop.get(k)!);
          probs.set(k, estp);
          
          
          if ( tol <= 0 || (estp > tol) && !finished.has(pt2str(k))) {
            // console.log(`${queue.has( k )? "Re-a" : "A"}dding child!` , k, ':\t ', nodenumber(k), probs.get(k));
            children_enqueued  = true;
            if ( !queue.has( k ) ) {
              queue.set(k, NEW);
            }
          }
        }
        
        if (children_enqueued) {
          queue.set(current_pt, children_enqueued ? NEW : "merge");
        }
        else {
          finished.add(pt2str(current_pt));
        }
      }
      else {
        // console.log(`skipped!\t queue size: ${queue.size()}\t maxQLen: ${maxQLen} \t p: ${probs.get(current_pt)}\t tol: ${tol}`);
      }
      // console.log(best.get(START), current_pt, (best.get(START)!.getOr(current_pt, 1)));

    }
  }

  // console.log('****************************************');
  let finalDist = best.get(START)!;
  finalDist.keyList().forEach( k => console.log('   ', k, finalDist.get(k)))
  // console.log(best);
}
function evalProg(prog: bril.Program) {
  let buffer : any[][] = [];

  for (let func of prog.functions) {
    if (func.name === "main") {
      evalFunc(func, buffer, 0, 0.0000001);
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
// process.argv.forEach((val, index) => {
  // console.log(`${index}: ${val}`);
// });
// console.log("hi", process.argv)

main();
