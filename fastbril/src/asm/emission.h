#ifndef EMISSION_H
#define EMISSION_H
#include "../bril-insns/instrs.h"
#include <stdio.h>

/**
 * emit the program prog as assembly to stream.
 * src_file is the original file name, if available
 */
void emit_program(FILE *stream, const char *src_file, program_t *prog);

#endif
