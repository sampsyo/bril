#ifndef INTERP_H
#define INTERP_H
#include <stdint.h>
#include "../bril-insns/instrs.h"

/**
 * internal representation of bril types.
 * we represent bools as ints, which is why they aren't here.
 * this has the convenience of all types being the same size: 64 bits
 */
typedef union value
{
  int64_t int_val;
  double float_val;
  union value* ptr_val;
} value_t;

/**
 * interpret one function with args.
 * dyn_insns is a pointer to a value which contains the number of
 * instructions executed so far.
 */
value_t interp_fun(program_t *prog, size_t *dyn_insns,
		   size_t which_fun, value_t *args, size_t num_args);

/**
 * interpret the main function of prog. count_insns -> we keep track of dynamic
 *  instruction count
 */
void interp_main(program_t *prog, value_t *args, size_t num_args, bool count_insns);

#endif
