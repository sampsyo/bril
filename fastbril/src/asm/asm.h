#ifndef ASM_H
#define ASM_H

#define __ARM_ARCH //x86 debugging

#ifdef __ARM_ARCH
#include "armv8.h"
#endif

typedef struct asm_func
{
  char name[128];
  size_t num_insns;
  size_t num_temps;
  size_t num_args;
  uint16_t *arg_types;
  uint16_t ret_tp;
  asm_insn_t *insns;
} asm_func_t;

typedef struct asm_prog
{
  size_t num_funcs;
  asm_func_t *funcs;
} asm_prog_t;


void emit_insns(FILE *stream, asm_prog_t *program);

void free_asm_prog(asm_prog_t prog);

#define INSN_SIZE  sizeof(asm_insn_t)
#endif


