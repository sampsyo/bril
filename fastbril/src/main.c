#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bril-insns/instrs.h"
#include "byte-io.h"
#include "interp/interp.h"
#include "libs/json.h"
#include "parser.h"
#include "pretty-printer.h"
// #include "emission.h"
#include "asm/asm.h"
#include "asm/linear-scan.h"
#include "asm/to_abstract_asm.h"
#include "asm/trivial-regalloc.h"

/* Bit masks for cmd flags/modes */
#define OUTPUT_BYTECODE 0x0001
#define COUNT_INSNS 0x0002
#define NO_INTERPRET 0x0004
#define PRINT_OUT 0x0008
#define EMIT_ASM 0x0010
#define READ_BYTECODE 0x0020

/**
 * read the contents of stdin and return a single heap allocated string.
 */
char *get_stdin()
{
  size_t buf_len = 128;
  char *buffer   = malloc(buf_len);
  size_t i       = 0;
  while (true)
    {
      if (i == buf_len - 1)
        {
          buf_len *= 2;
          buffer = realloc(buffer, buf_len);
        }
      int c = getchar();
      if (c == EOF)
        {
          buffer[i] = 0x00;
          break;
        }
      buffer[i] = c;
      ++i;
    }
  return buffer;
}

/**
 * turn a string into a value_t that can be used by the interpreter
 */
value_t parse_argument(const char *str, briltp expected)
{
  switch (expected)
    {
    case BRILINT:
      return (value_t){.int_val = strtol(str, 0, 0)};
    case BRILBOOL:
      if (strcmp(str, "true") == 0)
        return (value_t){.int_val = 1};
      else
        return (value_t){.int_val = 0};
    case BRILFLOAT:
      return (value_t){.float_val = strtod(str, 0)};
    default:
      return (value_t){.int_val = 0};
    }
}

int main(int argc, char **argv)
{
  long options    = 0;
  char *bout_file = 0, *out_file = 0;
  char *args_strs[argc];
  size_t argidx = 0;
  for (int i = 1; i < argc; ++i)
    {
      if (strcmp(argv[i], "-p") == 0)
        options |= COUNT_INSNS;
      else if (strcmp(argv[i], "-b") == 0)
        options |= READ_BYTECODE;
      else if (strcmp(argv[i], "-bo") == 0)
        {
          options |= OUTPUT_BYTECODE;
          bout_file = i + 1 < argc ? argv[++i] : 0;
        }
      else if (strcmp(argv[i], "-pr") == 0)
        options |= PRINT_OUT;
      else if (strcmp(argv[i], "-ni") == 0)
        options |= NO_INTERPRET;
      else if (strcmp(argv[i], "-e") == 0)
        {
          options |= EMIT_ASM;
          out_file = i + 1 < argc ? argv[++i] : 0;
        }
      else
        {
          args_strs[argidx++] = argv[i];
        }
    }
  program_t *prog;
  char *string              = 0;
  struct json_value_s *root = 0;
  if (options & READ_BYTECODE)
    prog = read_program(stdin);
  else
    {
      string                          = get_stdin();
      root                            = json_parse(string, strlen(string));
      struct json_object_s *functions = root->payload;
      prog                            = parse_program(functions);
    }
  if (options & OUTPUT_BYTECODE)
    {
      FILE *f = fopen(bout_file ? bout_file : "my-output", "w+");
      output_program(prog, f);
      fclose(f);
    }
  if (!(options & NO_INTERPRET))
    {
      value_t args[argidx];
      briltp *tps = get_main_types(prog);
      if (!tps)
        return 1;
      for (size_t i = 0; i < argidx; ++i)
        args[i] = parse_argument(args_strs[i], tps[i]);

      interp_main(prog, args, argidx, options & COUNT_INSNS);
    }
  if (options & PRINT_OUT)
    format_program(stdout, prog);
  if (options & EMIT_ASM)
    {
      FILE *f           = fopen(out_file ? out_file : "output.s", "w+");
      asm_prog_t p      = bytecode_to_abs_asm(prog);
      asm_prog_t allocd = triv_allocate(p);
      free_asm_prog(p);
      emit_insns(f, &allocd);
      fclose(f);
    }
  free(string);
  free(root);
  free_program(prog);
  return 0;
}
