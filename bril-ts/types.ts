import * as bril from './bril.ts';

/**
 * An abstract type signature.
 *
 * Describes the shape and types of all the ingredients for a Bril operation
 * instruction: arguments, result, labels, and functions.
 */
export interface BaseSignature<T> {
  /**
   * The types of each argument to the operation.
   */
  args: T[],

  /**
   * The result type, if non-void.
   */
  dest?: T,

  /**
   * The number of labels required for the operation.
   */
  labels?: number,

  /**
   * The number of function names required for the operation.
   */
  funcs?: number,
}

/**
 * The concrete type signature for an operation.
 */
export type Signature = BaseSignature<bril.Type>;

/**
 * A polymorphic type variable.
 */
export type TVar = {"tv": string};

/**
 * Like bril.Type, except that type variables may occur at the leaves.
 */
export type PolyType = bril.PrimType | TVar | {"ptr": PolyType};

/**
 * A polymorphic type signature, universally quantified over a single
 * type variable.
 */
export interface PolySignature {
    tvar: TVar;
    sig: BaseSignature<PolyType>;
}

/**
 * The type of a Bril function.
 */
export interface FuncType {
  /**
   * The types of the function's arguments.
   */
  args: bril.Type[],

  /**
   * The function's return type.
   */
  ret: bril.Type | undefined,
}

/**
 * Type signatures for the Bril operations we know.
 */
export const OP_SIGS: {[key: string]: Signature | PolySignature} = {
  // Core.
  'add': {args: ['int', 'int'], dest: 'int'},
  'mul': {args: ['int', 'int'], dest: 'int'},
  'sub': {args: ['int', 'int'], dest: 'int'},
  'div': {args: ['int', 'int'], dest: 'int'},
  'eq': {args: ['int', 'int'], dest: 'bool'},
  'lt': {args: ['int', 'int'], dest: 'bool'},
  'gt': {args: ['int', 'int'], dest: 'bool'},
  'le': {args: ['int', 'int'], dest: 'bool'},
  'ge': {args: ['int', 'int'], dest: 'bool'},
  'not': {args: ['bool'], dest: 'bool'},
  'and': {args: ['bool', 'bool'], dest: 'bool'},
  'or': {args: ['bool', 'bool'], dest: 'bool'},
  'jmp': {args: [], 'labels': 1},
  'br': {args: ['bool'], 'labels': 2},
  'id': {tvar: {tv: 'T'}, sig: {args: [{tv: 'T'}], dest: {tv: 'T'}}},
  'nop': {args: []},

  // Floating point.
  'fadd': {args: ['float', 'float'], dest: 'float'},
  'fmul': {args: ['float', 'float'], dest: 'float'},
  'fsub': {args: ['float', 'float'], dest: 'float'},
  'fdiv': {args: ['float', 'float'], dest: 'float'},
  'feq': {args: ['float', 'float'], dest: 'bool'},
  'flt': {args: ['float', 'float'], dest: 'bool'},
  'fgt': {args: ['float', 'float'], dest: 'bool'},
  'fle': {args: ['float', 'float'], dest: 'bool'},
  'fge': {args: ['float', 'float'], dest: 'bool'},

  // Memory.
  'alloc': {tvar: {tv: 'T'}, sig: {args: ['int'], dest: {ptr: {tv: 'T'}}}},
  'free': {tvar: {tv: 'T'}, sig: {args: [{ptr: {tv: 'T'}}]}},
  'store': {tvar: {tv: 'T'}, sig: {args: [{ptr: {tv: 'T'}}, {tv: 'T'}]}},
  'load': {tvar: {tv: 'T'}, sig: {args: [{ptr: {tv: 'T'}}], dest: {tv: 'T'}}},
  'ptradd': {tvar: {tv: 'T'}, sig: {args: [{ptr: {tv: 'T'}}, 'int'], dest: {ptr: {tv: 'T'}}}},

  // Speculation.
  'speculate': {args: []},
  'commit': {args: []},
  'guard': {args: ['bool'], labels: 1},

  // Character.
  'ceq': {args: ['char', 'char'], dest: 'bool'},
  'clt': {args: ['char', 'char'], dest: 'bool'},
  'cgt': {args: ['char', 'char'], dest: 'bool'},
  'cle': {args: ['char', 'char'], dest: 'bool'},
  'cge': {args: ['char', 'char'], dest: 'bool'},
  'char2int': {args: ['char'], dest: 'int'},
  'int2char': {args: ['int'], dest: 'char'},
};
