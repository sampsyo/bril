#include "emission.h"
#include <string.h>

/* macro system means we will emit x86 on x86 arch and arm on arm arch */
#ifdef __x86_64__

/* TODO Susan implement */
void emit_program(FILE *stream, const char *src_file, program_t *prog){}

#elif __ARM_ARCH

/**
 * convenience functions for common instructions
 */
static inline void load(FILE *stream, int reg, uint16_t dest)
{
  fprintf(stream, "\tldr\tx%d, [sp, %d]\n", reg, 16 + dest * 8);
}

static inline void fload(FILE *stream, int reg, uint16_t dest)
{
  fprintf(stream, "\tldr\td%d, [sp, %d]\n", reg, 16 + dest * 8);
}

static inline void store(FILE *stream, int reg, uint16_t dest)
{
   fprintf(stream, "\tstr\tx%d, [sp, %d]\n", reg, 16 + dest * 8);
}

static inline void fstore(FILE *stream, int reg, uint16_t dest)
{
   fprintf(stream, "\tstr\td%d, [sp, %d]\n", reg, 16 + dest * 8);
}

static inline void emit(FILE *stream, const char *opcode, int dest, int a1, int a2)
{
  fprintf(stream, "\t%s\tx%d, x%d, x%d\n", opcode, dest, a1, a2);
}

static inline void femit(FILE *stream, const char *opcode, int dest, int a1, int a2)
{
  fprintf(stream, "\t%s\td%d, d%d, d%d\n", opcode, dest, a1, a2);
}


static inline void mov(FILE *stream, const char *dest, const char *src)
{
  fprintf(stream, "\tmov\t%s, %s\n", dest, src);
}


static inline int max(int a, int b)
{
  return a > b ? a : b;
}

/**
 * translate bril opcode to armv8 float compare flag
 */
static inline char* fpcflag(uint16_t opcode)
{
  switch(opcode)
    {
    case FEQ: return "eq";
    case FLT: return "mi";
    case FLE: return "ls";
    case FGT: return "gt";
    case FGE: return "ge";
    }
}

/**
 * emits asm to store dest <- value
 */
void emit_constant(FILE *stream, uint16_t dest, int64_t value)
{
  fprintf(stream, "\tmov\tx0, %ld\n", value & 0xffff);
  int64_t tmp = value >> 16;
  for(size_t i = 1; i < 4; ++i)
    {
      /* if(tmp == 0 || tmp == -1) */
      /* 	break; */
      fprintf(stream, "\tmovk\tx0, %ld, lsl %ld\n", tmp & 0xffff, i * 16);
      tmp = tmp >> 16;
    }
  store(stream, 0, dest);
}

/**
 * iterate over the instructions of the function of prog at index
 * which_fun, and emit the instructions for the body. Does not do
 * any prologue/epilogue
 */
void emit_instructions(FILE *stream, program_t *prog, size_t which_fun)
{
  function_t f = prog->funcs[which_fun];
  for(size_t i = 0; i < f.num_insns; ++i)
    {
      instruction_t *insn = f.insns + i;
      if(is_labelled(*insn))
	fprintf(stream, ".LF%s%lx:\n", f.name, i);
      uint16_t opcode = get_opcode(*insn);
      switch(opcode)
	{
	case NOP:
	  fprintf(stream, "\tnop\n");
	  break;
	case CONST:
	  emit_constant(stream, insn->const_insn.dest, insn->const_insn.value);
	  break;
	case LCONST:
	  emit_constant(stream, insn->long_const_insn.dest,
			(insn + 1)->const_ext.int_val);
	  ++i;
	  break;
	case ADD:
	case MUL:
	case SUB:
	case AND:
	  load(stream, 0, insn->norm_insn.arg1);
	  load(stream, 1, insn->norm_insn.arg2);
	  emit(stream, opcode_to_string(opcode), 0, 0, 1);
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	case OR:
	  load(stream, 0, insn->norm_insn.arg1);
	  load(stream, 1, insn->norm_insn.arg2);
	  emit(stream, "orr", 0, 0, 1);
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	case DIV:
	  load(stream, 0, insn->norm_insn.arg1);
	  load(stream, 1, insn->norm_insn.arg2);
	  emit(stream, "sdiv", 0, 0, 1);
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	case NOT:
	  load(stream, 0, insn->norm_insn.arg1);
	  fprintf(stream, "\tcmp\tx0, 0\n");
	  fprintf(stream, "\tcset\tx0, eq\n");
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	case EQ:
	case LT:
	case GT:
	case LE:
	case GE:
	  load(stream, 0, insn->norm_insn.arg1);
	  load(stream, 1, insn->norm_insn.arg2);
	  fprintf(stream, "\tcmp\tx0, x1\n");
	  fprintf(stream, "\tcset\tx0, %s\n", opcode_to_string(opcode));
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	case FADD:
	case FMUL:
	case FSUB:
	case FDIV:
	  fload(stream, 0, insn->norm_insn.arg1);
	  fload(stream, 1, insn->norm_insn.arg2);
	  femit(stream, opcode_to_string(opcode), 0, 0, 1);
	  fstore(stream, 0, insn->norm_insn.dest);
	  break;
	case FEQ:
	case FLT:
	case FLE:
	case FGT:
	case FGE:
	  fload(stream, 0, insn->norm_insn.arg1);
	  fload(stream, 1, insn->norm_insn.arg2);
	  fprintf(stream, "\tfcmpe\td0, d1\n");
	  fprintf(stream, "\tcset\tx0, %s\n", fpcflag(opcode));
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	case PRINT:
	  {
	    uint16_t num_args = insn->print_insn.num_prints;
	    i += num_args / 2;
	    uint16_t *args = &insn->print_insn.type1;
	    for(uint16_t j = 0; j < num_args * 2; j += 2)
	      {
		if(j != 0)
		  {
		    fprintf(stream, "\tmov\tw0, 32\n");
		    fprintf(stream, "\tbl\tputchar\n");
		  }
		switch(args[j])
		  {
		  case BRILINT:
		    load(stream, 0, args[j + 1]);
		    fprintf(stream, "\tbl\tpriint\n");
		    break;
		  case BRILBOOL:
		    load(stream, 0, args[j + 1]);
		    fprintf(stream, "\tbl\tpribool\n");
		    break;
		  case BRILFLOAT:
		    fload(stream, 0, args[j + 1]);
		    fprintf(stream, "\tbl\tprifloat\n");
		    break;
		  }
	      }
	    fprintf(stream, "\tmov\tw0, 10\n");
	    fprintf(stream, "\tbl\tputchar\n");
	  }
	  break;
	case CALL:
	  {
	    uint16_t num_args = insn->call_inst.num_args;
	    uint16_t *args = (uint16_t*) (insn + 1);
	    function_t *fun = prog->funcs + insn->call_inst.target;
	    int norm_args = 0, float_args = 0;
	    for(size_t i = 0; i < fun->num_args; ++i)
	      {
		if(fun->arg_types[i] == BRILFLOAT)
		  ++float_args;
		else ++norm_args;
	      }
	    int norm_stack_needed = max((norm_args - 9) * 8, 0);
	    int float_stack_needed = max((float_args - 8) * 8, 0);
	    int stack_needed = norm_stack_needed + float_stack_needed;
	    if(stack_needed % 16 != 0) stack_needed += 8;
	    if(stack_needed)
	      fprintf(stream, "\tsub\tsp, sp, %d\n", stack_needed);
	    int norm_args_left = norm_args;
	    int float_args_left = float_args;
	    if(num_args != fun->num_args)
	      {
		fprintf(stderr, "Arg number mismatch calling %s. Found %d, \
expected %ld.Exiting.\n", fun->name, num_args, fun->num_args);
		exit(1);
	      }
	    int stack_args_left = max(norm_args - 9, 0) + max(float_args - 8, 0) - 1;
	    for(int argidx = num_args - 1; argidx >= 0; --argidx)
	      {
		if(fun->arg_types[argidx] == BRILFLOAT)
		  {
		    if(float_args_left > 8)
		      {
			fload(stream, 0, args[argidx] + stack_needed / 8);
			fprintf(stream, "\tstr\td0, [sp, %d]\n", stack_args_left * 8);
		      } else
		      {
			fload(stream, float_args_left - 1,
			      args[argidx] + stack_needed / 8);
		      }
		    --float_args_left;
		  } else
		  {
		    if(norm_args_left > 9)
		      {
			load(stream, 0, args[argidx] + stack_needed / 8);
			fprintf(stream, "\tstr\tx0, [sp, %d]\n", stack_args_left * 8);
		      } else
		      {
			load(stream, norm_args_left - 1,
			     args[argidx] + stack_needed / 8);
		      }
		    --norm_args_left;
		  }
		--stack_args_left;
	      }
	    fprintf(stream, "\tbl\t%s\n",
		    prog->funcs[insn->call_inst.target].name);
	    if(stack_needed)
	      fprintf(stream, "\tadd\tsp, sp, %d\n", stack_needed);
	    if(fun->ret_tp == BRILFLOAT)
	      fstore(stream, 0, insn->call_inst.dest);
	    else if(fun->ret_tp != BRILVOID)
	      store(stream, 0, insn->call_inst.dest);
	    i += (num_args + 3) / 4;
	  }
	  break;
	case RET:
	  switch(f.ret_tp)
	    {
	    case BRILVOID:
	      break;
	    case BRILFLOAT:
	      fload(stream, 0, insn->norm_insn.arg1);
	      break;
	    default:
	      load(stream, 0, insn->norm_insn.arg1);
	    }
	  fprintf(stream, "\tb\t.L%s.ret\n", f.name);
	  break;
	case JMP:
	  fprintf(stream, "\tb\t.LF%s%x\n", f.name, insn->norm_insn.dest);
	  break;
	case BR:
	  load(stream, 0, insn->br_inst.test);
	  fprintf(stream, "\tcbz\tx0, .LF%s%x\n", f.name, insn->br_inst.lfalse);
	  fprintf(stream, "\tb\t.LF%s%x\n", f.name, insn->br_inst.ltrue);
	  break;
	case ID:
	  if(insn->norm_insn.arg2 == BRILFLOAT)
	    {
	      fload(stream, 0, insn->norm_insn.arg1);
	      fstore(stream, 0, insn->norm_insn.dest);
	    } else
	    {
	      load(stream, 0, insn->norm_insn.arg1);
	      store(stream, 0, insn->norm_insn.dest);
	    }
	  break;
	case ALLOC:
	  load(stream, 0, insn->norm_insn.arg1);
	  fprintf(stream, "\tlsl\tx0, x0, 3\n");
	  fprintf(stream, "\tbl\tmalloc\n");
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	case FREE:
	  load(stream, 0, insn->norm_insn.arg1);
	  fprintf(stream, "\tbl\tfree\n");
	  break;
	case STORE:
	  /* pointer */
	  load(stream, 0, insn->norm_insn.arg1);
	  /* value */
	  load(stream, 1, insn->norm_insn.arg2);
	  fprintf(stream, "\tstr\tx1, [x0]\n");
	  break;
	case LOAD:
	  /* pointer */
	  load(stream, 0, insn->norm_insn.arg1);
	  fprintf(stream, "\tldr\tx0, [x0]\n");
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	case PTRADD:
	  load(stream, 0, insn->norm_insn.arg1);
	  load(stream, 1, insn->norm_insn.arg2);
	  fprintf(stream, "\tadd\tx0, x0, x1, lsl 3\n");
	  store(stream, 0, insn->norm_insn.dest);
	  break;
	default:
	  fprintf(stderr, "unsupported opcode: %d\n", opcode);
	}
    }
  fprintf(stream, ".LF%s%lx:\n", f.name, f.num_insns);
}


/**
 * emit asm for which_fun in prog to stream.
 */
void emit_function(FILE *stream, program_t *prog, size_t which_fun)
{
  function_t f = prog->funcs[which_fun];
  size_t stack_offset = 8 * f.num_tmps + 8;
  bool is_main = strcmp(f.name, "main") == 0;
  if(is_main && f.num_args != 0)
    stack_offset += 8;
  if(stack_offset % 16 != 0)
    stack_offset += 8;
  fprintf(stream, "\t.global %s\n", f.name);
  fprintf(stream, "\t.type\t%s, %%function\n", f.name);
  fprintf(stream, "%s:\n", f.name);
  if(stack_offset < 512)
    fprintf(stream, "\tstp\tx29, x30, [sp, -%ld]!\n", stack_offset);
  else
    {
      fprintf(stream, "\tsub\tsp, sp, %ld\n", stack_offset);
      fprintf(stream, "\tstp\tx29, x30, [sp]\n");
    }
  mov(stream, "x29", "sp");
  if(is_main && f.num_args != 0)
    {
      fprintf(stream, "\tstr\tx19, [sp, %ld]\n", stack_offset - 16);
      mov(stream, "x19", "x1");
      for(size_t i = 0; i < f.num_args; ++i)
	{
	  fprintf(stream, "\tldr\tx0, [x19, %ld]\n", (i + 1) * 8);
	  switch(f.arg_types[i])
	    {
	    case BRILINT:
	      fprintf(stream, "\tbl\tint_of_string\n");
	      store(stream, 0, i);
	      break;
	    case BRILBOOL:
	      fprintf(stream, "\tbl\tbool_of_string\n");
	      store(stream, 0, i);
	      break;
	    case BRILFLOAT:
	      fprintf(stream, "\tbl\tfloat_of_string\n");
	      fstore(stream, 0, i);
	      break;
	    default:
	      fprintf(stderr, "main cannot have pointer arguments! Exiting.\n");
	      exit(1);
	    }
	}
    } else
    {
      size_t double_args = 0, other_args = 0, spilled_args = 0;
      /* args passed in memory */
      for(size_t i = 0; i < f.num_args; ++i)
	{
	  if(f.arg_types[i] == BRILFLOAT)
	    {
	      if(double_args < 7)
		++double_args;
	      else
		{
		  fprintf(stream, "\tldr\td9, [sp, %ld]\n",
			  stack_offset + 8 * spilled_args++);
		  fstore(stream, 9, ++double_args);
		}
	    } else
	    {
	      if(other_args < 7)
		++other_args;
	      else
		{
		  fprintf(stream, "\tldr\tx9, [sp, %ld]\n",
			  stack_offset + 8 * spilled_args++);
		  store(stream, 9, ++other_args);
		}
	    }
	}
      double_args = 0; other_args = 0;
      /* args passed in registers */
      for(size_t i = 0; i < f.num_args; ++i)
	{
	  if(f.arg_types[i] == BRILFLOAT)
	    {
	      if(double_args < 8)
		fstore(stream, double_args++, i);
	    } else
	    {
	      if(other_args < 8)
		store(stream, other_args++, i);
	    }
	}
    }
  emit_instructions(stream, prog, which_fun);

  fprintf(stream, ".L%s.ret:\n", f.name);
  if(is_main && f.num_args != 0)
    fprintf(stream, "\tldr\tx19, [sp, %ld]\n", stack_offset - 16);
  if(is_main && f.ret_tp == BRILVOID)
    mov(stream, "w0", "0");
  if(stack_offset < 512)
    fprintf(stream, "\tldp\tx29, x30, [sp], %ld\n", stack_offset);
  else
    {
      fprintf(stream, "\tldp\tx29, x30, [sp]\n");
      fprintf(stream, "\tadd\tsp, sp, %ld\n", stack_offset);
    }
  fprintf(stream, "\tret\n");
   }

void emit_program(FILE *stream, const char *src_file, program_t *prog)
{
  fprintf(stream, "\t.arch armv8-a\n");
  fprintf(stream, "\t.text\n");
  for(size_t i = 0; i < prog->num_funcs; ++i)
    emit_function(stream, prog, i);
}

#endif
