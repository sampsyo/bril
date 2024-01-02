#include "asm.h"

#include "../bril-insns/types.h"

#include <stdlib.h>

#define __ARM_ARCH //DEBUGGING ON X86!!!!
#ifdef __ARM_ARCH
#include "armv8.h"

static inline const char *op_to_string(norm_arm_op_t op)
{
  switch (op)
    {
    case AADD:  return "add";
    case AMUL:  return "mul";
    case ASUB:  return "sub";
    case AAND:  return "and";
    case AORR:  return "orr";
    case ASDIV: return "sdiv";
    case AFADD: return "fadd";
    case AFMUL: return "fmul";
    case AFSUB: return "fsub";
    case AFDIV: return "fdiv";
    case ALSL:  return "lsl";
    }
}

static inline const char *reg_to_string(arm_reg_t reg)
{
  switch(reg)
    {
    case SP: return "sp"; case X0: return "x0"; case X1: return "x1";
    case X2: return "x2"; case X3: return "x3"; case X4: return "x4";
    case X5: return "x5"; case X6: return "x6"; case X7: return "x7";
    case X8: return "x8"; case X9: return "x9"; case X10: return "x10";
    case X11: return "x11"; case X12: return "x12"; case X13: return "x13";
    case X14: return "x14"; case X15: return "x14"; case X16: return "x16";
    case X17: return "x17"; case X18: return "x18"; case X19: return "x19";
    case X20: return "x20"; case X21: return "x21"; case X22: return "x22";
    case X23: return "x23"; case X24: return "x24"; case X25: return "x25";
    case X26: return "x26"; case X27: return "x27"; case X28: return "x28";
    case X29: return "x29"; case X30: return "x30"; case XZR: return "xzr";
    case D0: return "d0"; case D1: return "d1"; case D2: return "d2";
    case D3: return "d3"; case D4: return "d4"; case D5: return "d5";
    case D6: return "d6"; case D7: return "d7"; case D8: return "d8";
    case D9: return "d9"; case D10: return "d10"; case D11: return "d11";
    case D12: return "d12"; case D13: return "d13"; case D14: return "d14";
    case D15: return "d15"; case D16: return "d16"; case D17: return "d17";
    case D18: return "d18"; case D19: return "d19"; case D20: return "d20";
    case D21: return "d21"; case D22: return "d22"; case D23: return "d23";
    case D24: return "d24"; case D25: return "d25"; case D26: return "d26";
    case D27: return "d27"; case D28: return "d28"; case D29: return "d29";
    case D30: return "d30"; case D31: return "d31";
    }
}

static inline void emit_arg(FILE *stream, arm_arg_tagged_t arg)
{
  switch(arg.type)
    {
    case REG: fprintf(stream, "%s", reg_to_string(arg.value.reg));
      break;
    case TMP: fprintf(stream, "t%d", arg.value.tmp);
      break;
    case CNST: fprintf(stream, "%d", arg.value.cnst);
    }
}

static inline const char *cmp_code_string(arm_cmp_flags_t flag)
{
  switch(flag)
    {
    case CMPEQ: return "eq";
    case CMPMI: return "mi";
    case CMPLS: return "ls";
    case CMPGT: return "gt";
    case CMPGE: return "ge";
    case CMPLT: return "lt";
    case CMPLE: return "le";
    }
}

void emit_arm_insns(FILE *stream, tagged_arm_insn_t *insns, size_t num_insns)
{
  for(size_t i = 0; i < num_insns; ++i)
    {
      switch(insns[i].type)
	{
	case ANORM:
	  fprintf(stream, "\t%s\t", op_to_string(insns[i].value.norm.op));
	  emit_arg(stream, insns[i].value.norm.dest);
	  fprintf(stream, ", ");
	  emit_arg(stream, insns[i].value.norm.a1);
	  fprintf(stream, ", ");
	  emit_arg(stream, insns[i].value.norm.a2);
	  if(insns[i].value.norm.lsl)
	    fprintf(stream, ", lsl #%d", insns[i].value.norm.lsl);
	  putc('\n', stream);
	  break;
	case ACMP:
	  if(insns[i].value.cmp.is_float)
	    fprintf(stream, "\tfcmpe\t");
	  else
	    fprintf(stream, "\tcmp\t");
	  emit_arg(stream, insns[i].value.cmp.a1);
	  fprintf(stream, ", ");
	  emit_arg(stream, insns[i].value.cmp.a2);
	  putc('\n', stream);
	  break;
	case ASET:
	  fprintf(stream, "\tcset\t");
	  emit_arg(stream, insns[i].value.set.dest);
	  fprintf(stream, ", %s\n", cmp_code_string(insns[i].value.set.flag));
	  break;
	case AMOV:
	  fprintf(stream, "\tmov\t");
	  emit_arg(stream, insns[i].value.mov.dest);
	  fprintf(stream, ", ");
	  emit_arg(stream, insns[i].value.mov.src);
	  putc('\n', stream);
	  break;
	case AMOVK:
	  fprintf(stream, "\tmovk\t");
	  emit_arg(stream, insns[i].value.movk.dest);
	  fprintf(stream, ", %d, lsl %d\n", insns[i].value.movk.val,
		  insns[i].value.movk.lsl);
	  break;
	case AMOVC:
	  fprintf(stream, "\tmovc\t");
	    emit_arg(stream, insns[i].value.movc.dest);
	  fprintf(stream, ", %ld\n", insns[i].value.movc.val);
	  break;
	case ACBZ:
	  fprintf(stream, "\tcbz\t");
	  emit_arg(stream, insns[i].value.cbz.cond);
	  fprintf(stream, ", %s\n", insns[i].value.cbz.dest);
	  break;
	case ACALL:
	  fprintf(stream, "\tbl\t%s\n", insns[i].value.call.name);
	  break;
	case ALDR:
	  fprintf(stream, "\tldr\t");
	  emit_arg(stream, insns[i].value.ldr.dest);
	  fprintf(stream, ", [");
	  emit_arg(stream, insns[i].value.ldr.address);
	  if(insns[i].value.ldr.offset)
	    fprintf(stream, ", %d", insns[i].value.ldr.offset);
	  fprintf(stream, "]\n");
	  break;
	case ASTR:
	  fprintf(stream, "\tstr\t");
	  emit_arg(stream, insns[i].value.str.value);
	  fprintf(stream, ", [");
	  emit_arg(stream, insns[i].value.str.address);
	  if(insns[i].value.str.offset)
	    fprintf(stream, ", %d", insns[i].value.str.offset);
	  fprintf(stream, "]\n");
	  break;
	case AABSCALL:
	  {
	    size_t num_args = insns[i].value.abs_call.num_args;
	    fprintf(stream, "\t");
	    if(insns[i].value.abs_call.ret_tp != BRILVOID)
	      fprintf(stream, "t%d <- ", insns[i].value.abs_call.dest);
	    fprintf(stream, "call\t%s\t", insns[i].value.abs_call.name);
	    for(size_t x = 0; x < num_args; x += 32)
	      {
		for(size_t argi = 0; x + argi < num_args && argi < 32; ++argi)
		  {
		    fprintf(stream, "t%d ",
			    insns[i + 1 + x/32].value.abs_call_ext.typed_temps[argi] &
			    0xffff);
		  }
		++i;
	      }
	    fprintf(stream, "\n");
	  }
	  break;
	case AABSEXT:
	  fprintf(stderr, "shouldn't be printing extension\n");
	  break;
	case AOTHER:
	  fprintf(stream, "%s\n", insns[i].value.other);
	  break;
	}
    }
}

void emit_function(FILE *stream, asm_func_t *f)
{
  fprintf(stream, "\t.global %s\n", f->name);
  fprintf(stream, "\t.type\t%s, %%function\n", f->name);
  fprintf(stream, "%s:\n", f->name);
  emit_arm_insns(stream, f->insns, f->num_insns);
}

void emit_insns(FILE *stream, asm_prog_t *prog)
{
  fprintf(stream, "\t.arch armv8-a\n\t.text\n");
  for(size_t i = 0; i < prog->num_funcs; ++i)
    emit_function(stream, prog->funcs + i);
}

#else

void emit_insns(FILE *stream, asm_prog_t *program)
{
  fprintf(stderr, "architecture not supported!\n");
  exit(1);
}

#endif


void free_func(asm_func_t *f)
{
  free(f->insns);
  free(f->arg_types);
}

void free_asm_prog(asm_prog_t prog)
{
  for(size_t i = 0; i < prog.num_funcs; ++i)
    free_func(prog.funcs + i);
  free(prog.funcs);
}
