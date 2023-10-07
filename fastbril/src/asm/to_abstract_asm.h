#ifndef TO_ABS_H
#define TO_ABS_H

#include "../bril-insns/instrs.h"
#include "asm.h"


//#define __ARM_ARCH
#ifdef __ARM_ARCH
#include "armv8.h"
#endif

asm_prog_t bytecode_to_abs_asm(program_t *prog);


#endif
