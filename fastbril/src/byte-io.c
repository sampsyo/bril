#define _GNU_SOURCE
#include "byte-io.h"
#include <string.h>



void output_function(function_t *func, FILE *dest)
{
  fprintf(dest, "%s\n", func->name);
  fwrite(&func->num_args, sizeof(size_t), 1, dest);
  fwrite(func->arg_types, sizeof(briltp), func->num_args, dest);
  fwrite(&func->ret_tp, sizeof(briltp), 1, dest);
  fwrite(&func->num_insns, sizeof(size_t), 1, dest);
  fwrite(&func->num_tmps, sizeof(size_t), 1, dest);
  fwrite(func->insns, sizeof(instruction_t), func->num_insns, dest);
}

void output_program(program_t *prog, FILE *dest)
{
  //printf("%ld\n", prog->num_funcs);
  fwrite(&prog->num_funcs, sizeof(size_t), 1, dest);
  for(size_t i = 0; i < prog->num_funcs; ++i)
    {
      output_function(&prog->funcs[i], dest);
    }
}

void read_function(function_t *dest, FILE *source)
{
  char *name = 0;
  size_t len = 0;
  len = getline(&name, &len, source);
  name[len - 1] = 0;
  dest->name = name;
  fread(&dest->num_args, sizeof(size_t), 1, source);
  dest->arg_types = malloc(sizeof(briltp) * dest->num_args);
  fread(dest->arg_types, sizeof(briltp), dest->num_args, source);
  fread(&dest->ret_tp, sizeof(briltp), 1, source);
  fread(&dest->num_insns, sizeof(size_t), 1, source);
  fread(&dest->num_tmps, sizeof(size_t), 1, source);
  instruction_t *insns = malloc(sizeof(instruction_t) * dest->num_insns);
  fread(insns, sizeof(instruction_t), dest->num_insns, source);
  dest->insns = insns;
}


program_t *read_program(FILE *source)
{
  size_t num_funcs;
  fread(&num_funcs, sizeof(size_t), 1, source);
  program_t *prog = malloc(sizeof(program_t) + sizeof(function_t) * num_funcs);
  prog->num_funcs = num_funcs;
  for(size_t i = 0; i < num_funcs; ++i)
    read_function(prog->funcs + i, source);
  return prog;
}
