#ifndef PRETTY_PRINTER_H
#define PRETTY_PRINTER_H
#include "bril-insns/instrs.h"
#include <stdio.h>

/**
 * formats the instruction insns[idx] to stream.
 * needs the program struct to resolve some naming stucc
 */
size_t format_insn(FILE *stream, program_t *prog, instruction_t *insns, size_t idx);

/**
 * pretty prints a program
 */
void format_program(FILE *stream, program_t *prog);

#endif
