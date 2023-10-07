#ifndef BYTE_OUTPUT_H
#define BYTE_OUTPUT_H
#include "bril-insns/instrs.h"
#include <stdio.h>

/**
 * emit prog as bytecode to dest
 */
void output_program(program_t *prog, FILE *dest);

/**
 * read source as bytecode and return a program
 */
program_t *read_program(FILE *source);

#endif
