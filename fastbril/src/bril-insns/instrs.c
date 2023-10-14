#include "instrs.h"
#include "stdio.h"
#include "string.h"

inline uint16_t get_opcode(const instruction_t i)
{
  return i.norm_insn.opcode_lbled & 0x7fff;
}
inline bool is_labelled(const instruction_t i)
{
  return i.norm_insn.opcode_lbled & 0x8000;
}

void free_program(program_t *prog)
{
  for (size_t i = 0; i < prog->num_funcs; ++i)
    {
      free(prog->funcs[i].name);
      free(prog->funcs[i].insns);
      free(prog->funcs[i].arg_types);
    }
  free(prog);
}

char type_to_char[5] = {'i', 'b', 'f', 'p', 'v'};

#define TEST_OP(s, o)                                                          \
  if (o == op)                                                                 \
    {                                                                          \
      return s;                                                                \
    }
/**
 * this COULD be an array lookup, but this option gives more flexibility,
 * and the performance hit is only in parsing
 */
char *opcode_to_string(uint16_t op)
{
  TEST_OP("nop", NOP);
  TEST_OP("const", CONST);
  TEST_OP("add", ADD);
  TEST_OP("mul", MUL);
  TEST_OP("mul", MUL);
  TEST_OP("sub", SUB);
  TEST_OP("div", DIV);
  TEST_OP("eq", EQ);
  TEST_OP("lt", LT);
  TEST_OP("gt", GT);
  TEST_OP("le", LE);
  TEST_OP("ge", GE);
  TEST_OP("not", NOT);
  TEST_OP("and", AND);
  TEST_OP("or", OR);
  TEST_OP("jmp", JMP);
  TEST_OP("br", BR);
  TEST_OP("call", CALL);
  TEST_OP("ret", RET);
  TEST_OP("print", PRINT);
  TEST_OP("phi", PHI);
  TEST_OP("alloc", ALLOC);
  TEST_OP("free", FREE);
  TEST_OP("store", STORE);
  TEST_OP("load", LOAD);
  TEST_OP("ptradd", PTRADD);
  TEST_OP("fadd", FADD);
  TEST_OP("fmul", FMUL);
  TEST_OP("fsub", FSUB);
  TEST_OP("fdiv", FDIV);
  TEST_OP("feq", FEQ);
  TEST_OP("flt", FLT);
  TEST_OP("fle", FLE);
  TEST_OP("fgt", FGT);
  TEST_OP("fge", FGE);
  TEST_OP("id", ID);
  return "";
}

uint16_t ptr_depth(briltp tp) { return tp >> 2; }

uint16_t base_type(briltp tp) { return tp & 0b11; }

briltp *get_main_types(program_t *prog)
{
  for (size_t i = 0; i < prog->num_funcs; ++i)
    {
      if (strcmp(prog->funcs[i].name, "main") == 0)
        {
          return prog->funcs[i].arg_types;
        }
    }
  fprintf(stderr, "no main function found! exiting.\n");
  return 0;
}
