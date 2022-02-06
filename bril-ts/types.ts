import * as bril from './bril';

/**
 * The type signature for an operation.
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

export type Signature = BaseSignature<bril.Type>;

export type TVar = string;

export interface PolySignature {
    tvar: TVar;
    sig: BaseSignature<bril.Type | TVar>;
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
  'id': {tvar: 'T', sig: {args: ['T'], dest: 'T'}},

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
};
