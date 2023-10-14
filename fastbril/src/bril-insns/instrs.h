#ifndef INSTRS_H
#define INSTRS_H
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include "base.h"
#include "float.h"
#include "mem.h"
#include "ssa.h"
#include "types.h"
/**
 * see the documentation for instruction layouts
 */

typedef struct norm_instruction
{
  uint16_t opcode_lbled;
  uint16_t dest;
  uint16_t arg1;
  uint16_t arg2;
} norm_instruction_t;

typedef struct br_inst
{
  uint16_t opcode_lbled;
  uint16_t test;
  uint16_t ltrue;
  uint16_t lfalse;
} br_inst_t;

typedef struct call_inst
{
  uint16_t opcode_lbled;
  uint16_t dest;
  uint16_t num_args;
  uint16_t target;
} call_inst_t;

typedef struct call_args
{
  uint16_t args[4];
} call_args_t;

typedef struct phi_inst
{
  uint16_t opcode_lbled;
  uint16_t dest;
  uint16_t num_choices;
  uint16_t __unused;
} phi_inst_t;

typedef struct phi_extension
{
  uint16_t lbl1;
  uint16_t val1;
  uint16_t lbl2;
  uint16_t val2;
} phi_extension_t;

typedef struct const_instr
{
  uint16_t opcode_lbled;
  uint16_t dest;
  int32_t value;
} const_instr_t;

typedef struct long_const_instr
{
  uint16_t opcode_lbled;
  uint16_t dest;
  uint16_t type;
  uint16_t __unused;
} long_const_instr_t;

typedef union const_extn
{
  int64_t int_val;
  double float_val;
} const_extn_t;

typedef struct print_instr
{
  uint16_t opcode_lbled;
  uint16_t num_prints;
  uint16_t type1;
  uint16_t arg1;
} print_instr_t;

typedef struct print_args
{
  uint16_t type1;
  uint16_t arg1;
  uint16_t type2;
  uint16_t arg2;
} print_args_t;

typedef union instruction
{
  norm_instruction_t norm_insn;
  br_inst_t br_inst;
  phi_inst_t phi_inst;
  phi_extension_t phi_ext;
  const_instr_t const_insn;
  long_const_instr_t long_const_insn;
  const_extn_t const_ext;
  print_instr_t print_insn;
  print_args_t print_args;
  call_inst_t call_inst;
  call_args_t call_args;
} instruction_t;



typedef uint16_t briltp;

typedef struct function
{
  char *name;
  size_t num_args;
  briltp *arg_types;
  briltp ret_tp;
  size_t num_insns;
  size_t num_tmps;
  instruction_t *insns;
} function_t;


typedef struct program
{
  size_t num_funcs;
  function_t funcs[];
} program_t;

void free_program(program_t *prog);

/**
 * convenience functions for bit fiddling
 */
uint16_t get_opcode(const instruction_t);
bool is_labelled(const instruction_t i);

/**
 * take an encoded opcode and go back to the original string representation
 */
char *opcode_to_string(uint16_t);


uint16_t ptr_depth(briltp);
uint16_t base_type(briltp);

briltp *get_main_types(program_t *prog);

extern char type_to_char[];



#endif
