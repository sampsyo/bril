#include "pretty-printer.h"
#include <string.h>
#include <ctype.h>

#define TEST_OP(s, o) if(o == op) {return s;}

/**
 * pretty prints type to stream
 */
void format_type(FILE *stream, uint16_t type)
{
  uint16_t depth = ptr_depth(type);
  uint16_t base_tp = base_type(type);
  for(size_t i = 0; i < depth; ++i)
    fprintf(stream, "ptr<");
  switch(base_tp)
    {
    case BRILINT:
      fprintf(stream, "int");
      break;
    case BRILBOOL:
      fprintf(stream, "bool");
      break;
    case BRILFLOAT:
      fprintf(stream, "float");
      break;
    case BRILVOID:
      break;
    }
  for(size_t i = 0; i < depth; ++i)
    putc('>', stream);
}

/**
 * formats a function name to stream
 */
void format_fun_name(FILE *stream, const char *fun_name)
{
  putc('@', stream);
  char *num = strrchr(fun_name, '_');
  if(num)
    {
      for(const char *c = fun_name; c != num; ++c)
	putc(*c, stream);
    }
  else
    fprintf(stream, "%s", fun_name);
}


size_t format_insn(FILE *stream, program_t *prog, instruction_t *insns, size_t idx)
{
  if(is_labelled(insns[idx]))
    fprintf(stream, ".L%ld:\n", idx);
  switch(get_opcode(insns[idx]))
    {
    case CONST:
      fprintf(stream, "    t%d = const %d;\n", insns[idx].const_insn.dest,
	      insns[idx].const_insn.value);
      break;
    case ADD:
    case MUL:
    case SUB:
    case DIV:
    case EQ:
    case LT:
    case GT:
    case LE:
    case GE:
    case AND:
    case OR:
    case PTRADD:
    case FADD:
    case FMUL:
    case FSUB:
    case FDIV:
    case FEQ:
    case FLT:
    case FLE:
    case FGT:
    case FGE:
      fprintf(stream, "    t%d = %s t%d t%d;\n", insns[idx].norm_insn.dest,
	      opcode_to_string(get_opcode(insns[idx])),
	      insns[idx].norm_insn.arg1, insns[idx].norm_insn.arg2);
      break;
    case NOT:
    case ID:
    case ALLOC:
    case LOAD:
      fprintf(stream, "    t%d = %s t%d;\n", insns[idx].norm_insn.dest,
	      opcode_to_string(get_opcode(insns[idx])),
	      insns[idx].norm_insn.arg1);
      break;
    case JMP:
      fprintf(stream, "    jmp .L%d;\n", insns[idx].norm_insn.dest);
      break;
    case BR:
      fprintf(stream, "    br t%d .L%d .L%d;\n", insns[idx].br_inst.test,
	      insns[idx].br_inst.ltrue, insns[idx].br_inst.lfalse);
      break;
    case CALL:
      {
	const function_t *target = &prog->funcs[insns[idx].call_inst.target];
	if(target->ret_tp != BRILVOID)
	  {
	    fprintf(stream, "    t%d :", insns[idx].call_inst.dest);
	    format_type(stream, target->ret_tp);
	    fprintf(stream, " = call ");
	    format_fun_name(stream, target->name);
	  }
	else
	  {
	    fprintf(stream, "    call ");
	    format_fun_name(stream, target->name);
	  }
	uint16_t *args = (uint16_t*) (insns + idx + 1);
	for(size_t i = 0; i < insns[idx].call_inst.num_args; ++i)
	  fprintf(stream, " t%d", args[i]);
	fprintf(stream, ";\n");
	return idx + 1 + (insns[idx].call_inst.num_args + 3) / 4;
    }
    case RET:
      if(insns[idx].norm_insn.arg1 == 0xffff)
	fprintf(stream, "    ret;\n");
      else
	fprintf(stream, "    ret t%d;\n", insns[idx].norm_insn.arg1);
      break;
    case PRINT:
      fprintf(stream, "    print");
      uint16_t *args = (uint16_t*) &insns[idx].print_insn.arg1;
      for(size_t i = 0; i < insns[idx].print_insn.num_prints; ++i)
	fprintf(stream, " t%d", args[2 * i]);
      fprintf(stream, ";\n");
      return idx + 1 + insns[idx].print_insn.num_prints / 2;
    case LCONST:
      fprintf(stream, "    t%d = const ", insns[idx].long_const_insn.dest);
      switch(insns[idx].long_const_insn.type)
	{
	case BRILINT:
	  fprintf(stream, "%ld;\n", insns[idx + 1].const_ext.int_val);
	  break;
	case BRILFLOAT:
	  fprintf(stream, "%f;\n", insns[idx + 1].const_ext.float_val);
	  break;
	case BRILBOOL:
	  fprintf(stream, "%s;\n",
		  insns[idx + 1].const_ext.int_val ? "true" : "false");
	}
      return idx + 2;
    case PHI:
      fprintf(stream, "    t%d = phi", insns[idx].phi_inst.dest);
      uint16_t *phi_ext = (uint16_t*) (insns + idx + 1);
      for(size_t i = 0; i < insns[idx].phi_inst.num_choices; ++i)
	fprintf(stream, " t%d .L%d", phi_ext[2 * i + 1], phi_ext[2 * i]);
      fprintf(stream, ";\n");
      return idx + 1 + (insns[idx].phi_inst.num_choices + 1) / 2;
    case STORE:
      fprintf(stream, "    store t%d t%d;\n", insns[idx].norm_insn.arg1,
	      insns[idx].norm_insn.arg2);
      break;
    case FREE:
      fprintf(stream, "    free t%d;\n", insns[idx].norm_insn.arg1);
      break;
    }
  return idx + 1;
}

/**
 * formats the header of fun to stream
 */
void format_fun_header(FILE *stream, const function_t *fun)
{
  fprintf(stream, "@%s(", fun->name);
  for(size_t a = 0; a < fun->num_args; ++a)
    {
      if(a != 0)
	fprintf(stream, ", ");
      fprintf(stream, "t%ld :", a);
      format_type(stream, fun->arg_types[a]);
    }
  putc(')', stream);
  if(fun->ret_tp != BRILVOID)
    {
      fprintf(stream, " :");
      format_type(stream, fun->ret_tp);
    }
  putc('\n', stream);
}


void format_program(FILE *stream, program_t *prog)
{
  for(size_t f = 0; f < prog->num_funcs; ++f)
    {
      format_fun_header(stream, prog->funcs + f);
      fprintf(stream, "  {\n");
      size_t idx = 0;
      while(idx < prog->funcs[f].num_insns)
	idx = format_insn(stream, prog, prog->funcs[f].insns, idx);
      fprintf(stream, "  }\n\n");
    }
}
