/**
 * https://capra.cs.cornell.edu/bril/lang/memory.html
 */
// deno-lint-ignore no-empty-interface
export interface Pointer<T> {}
export function alloc<T>(size: bigint): Pointer<T>;
export function store<T>(pointer: Pointer<T>, value: T): void;
export function load<T>(pointer: Pointer<T>): T;
export function free<T>(pointer: Pointer<T>): void;
export function ptradd<T>(pointer: Pointer<T>, offset: bigint): Pointer<T>;
