#include "interp.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

value_t interpret_insn(program_t  *prog, size_t which_fun,
		       value_t *context, uint16_t *labels,
		       size_t *dyn_insns, size_t which_insn)
{
  while(which_insn < prog->funcs[which_fun].num_insns)
    {
      instruction_t *i = prog->funcs[which_fun].insns + which_insn;
      size_t next_insn = which_insn + 1;
      ++*dyn_insns;
      if(is_labelled(*i))
	{
	  labels[1] = labels[0];
	  labels[0] = which_insn;
	}
      switch(get_opcode(*i))
	{
	case CONST:
	  context[i->const_insn.dest] = (value_t) {.int_val = i->const_insn.value};
	  break;
	case ADD:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val +
	     context[i->norm_insn.arg2].int_val};
	  break;
	case MUL:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val *
	     context[i->norm_insn.arg2].int_val};
	  break;
	case SUB:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val -
	     context[i->norm_insn.arg2].int_val};
	  break;
	case DIV:
	  if(context[i->norm_insn.arg2].int_val == 0)
	    {
	      fprintf(stderr, "Divide by 0. Exiting\n");
	      exit(1);
	    }
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val /
	     context[i->norm_insn.arg2].int_val};
	  break;
	case EQ:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val ==
	     context[i->norm_insn.arg2].int_val ? 1 : 0};
	  break;
	case LT:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val <
	     context[i->norm_insn.arg2].int_val ? 1 : 0};
	  break;
	case GT:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val >
	     context[i->norm_insn.arg2].int_val ? 1 : 0};
	  break;
	case LE:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val <=
	     context[i->norm_insn.arg2].int_val ? 1 : 0};
	  break;
	case GE:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val >=
	     context[i->norm_insn.arg2].int_val ? 1 : 0};
	  break;
	case NOT:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = 1 ^ context[i->norm_insn.arg1].int_val};
	  break;
	case AND:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val &
	     context[i->norm_insn.arg2].int_val};
	  break;
	case OR:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].int_val |
	     context[i->norm_insn.arg2].int_val};
	  break;
	case JMP:
	  next_insn = i->norm_insn.dest;
	  break;
	case BR:
	  next_insn = context[i->br_inst.test].int_val == 1
	    ? i->br_inst.ltrue : i->br_inst.lfalse;
	  break;
	case CALL:
	  {
	    value_t args[i->call_inst.num_args];
	    for(size_t a = 0; a < i->call_inst.num_args; ++a)
	      args[a] = context[prog->funcs[which_fun].insns[which_insn + 1 + a / 4]
				.call_args.args[a % 4]];
	    next_insn += (i->call_inst.num_args + 3) / 4;
	    value_t tmp = interp_fun(prog, dyn_insns, i->call_inst.target,
				     args, i->call_inst.num_args);
	    if(i->call_inst.dest != 0xffff)
	      context[i->call_inst.dest] = tmp;
	    break;
	  }
	case RET:
	  if(i->norm_insn.arg1 == 0xffff)
	    return (value_t) {.int_val = 0xffffffffffffffff};
	  else
	    return context[i->norm_insn.arg1];
	case PRINT:
	  {
	    uint16_t *args = (uint16_t*) &(i->print_insn.type1);
	    for(size_t a = 0; a < i->print_insn.num_prints; ++a)
	      {
		if(a != 0)
		  printf(" ");
		switch(args[2 * a])
		  {
		  case BRILBOOL:
		    printf("%s", context[args[2 * a + 1]].int_val ? "true" : "false");
		    break;
		  case BRILINT:
		    printf("%ld", context[args[2 * a + 1]].int_val);
		    break;
		  case BRILFLOAT:
            {
              double f = context[args[2 * a + 1]].float_val;
              if (isnan(f)) {
                  printf("NaN");
              } else if (isinf(f)) {
                  if (f < 0) {
                      printf("-Infinity");
                  } else {
                      printf("Infinity");
                  }
              } else {
                  printf("%.17lf", f);
              }
		      break;
            }
		  default:
		    fprintf(stderr, "unrecognized type: %d. exiting.\n", args[2 * a]);
		    exit(1);
		  }
	      }
	    printf("\n");
	    next_insn += (i->print_insn.num_prints) / 2;
	    break;
	  }
	case LCONST:
	  context[i->long_const_insn.dest] =
	    *((value_t*) &prog->funcs[which_fun].insns[which_insn + 1]);
	  ++next_insn;
	  break;
	case NOP:
	  break;
	case ID:
	  context[i->norm_insn.dest] = context[i->norm_insn.arg1];
	  break;
	case ALLOC:
	  context[i->norm_insn.dest] = (value_t)
	    {.ptr_val = malloc(sizeof(value_t) * context[i->norm_insn.arg1].int_val)};
	  break;
	case FREE:
	  free(context[i->norm_insn.arg1].ptr_val);
	  break;
	case STORE:
	  *(context[i->norm_insn.arg1].ptr_val) = context[i->norm_insn.arg2];
	  break;
	case LOAD:
	  context[i->norm_insn.dest] = *(context[i->norm_insn.arg1].ptr_val);
	  break;
	case PTRADD:
	  context[i->norm_insn.dest] = (value_t)
	    {.ptr_val = context[i->norm_insn.arg1].ptr_val +
	     context[i->norm_insn.arg2].int_val};
	  break;
	case PHI:
	  {
	    size_t num_choices = i->phi_inst.num_choices;
	    next_insn += (num_choices + 1) / 2;
	    for(size_t a = 0; a < num_choices; ++a)
	      {
		if (labels[1] ==
		    ((uint16_t*) (&prog->funcs[which_fun]
				  .insns[which_insn + 1 + a/2].phi_ext))[(a % 2) * 2])
		  context[i->phi_inst.dest] =
		    context[
			    ((uint16_t*) (&prog->funcs[which_fun]
					  .insns[which_insn + 1 + a/2]
					  .phi_ext))[(a % 2) * 2 + 1]];
	      }
	    break;
	  }
	case FADD:
	   context[i->norm_insn.dest] = (value_t)
	    {.float_val = context[i->norm_insn.arg1].float_val +
	     context[i->norm_insn.arg2].float_val};
	   break;
	case FMUL:
	  context[i->norm_insn.dest] = (value_t)
	    {.float_val = context[i->norm_insn.arg1].float_val *
	     context[i->norm_insn.arg2].float_val};
	   break;
	case FSUB:
	   context[i->norm_insn.dest] = (value_t)
	    {.float_val = context[i->norm_insn.arg1].float_val -
	     context[i->norm_insn.arg2].float_val};
	   break;
	case FDIV:
	  context[i->norm_insn.dest] = (value_t)
	    {.float_val = context[i->norm_insn.arg1].float_val /
	     context[i->norm_insn.arg2].float_val};
	   break;
	case FEQ:
	   context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].float_val ==
	     context[i->norm_insn.arg2].float_val ? 1 : 0};
	   break;
	case FLT:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].float_val <
	     context[i->norm_insn.arg2].float_val ? 1 : 0};
	   break;
	case FLE:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].float_val <=
	     context[i->norm_insn.arg2].float_val ? 1 : 0};
	   break;
	case FGT:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].float_val >
	     context[i->norm_insn.arg2].float_val ? 1 : 0};
	   break;
	case FGE:
	  context[i->norm_insn.dest] = (value_t)
	    {.int_val = context[i->norm_insn.arg1].float_val >=
	     context[i->norm_insn.arg2].float_val ? 1 : 0};
	   break;
	default:
	  fprintf(stderr, "unrecognized opcode: %d, exiting.\n", get_opcode(*i));
	  exit(1);
	}
      which_insn = next_insn;
    }
  return (value_t) {.int_val = 0};
}


void interp_main(program_t *prog, value_t *args, size_t num_args, bool count_insns)
{
  for(size_t i = 0; i < prog->num_funcs; ++i)
    {
      if(strcmp(prog->funcs[i].name, "main") == 0)
	{
	  size_t dyn = 0;
	  interp_fun(prog, &dyn, i, args, num_args);
	  if(count_insns)
	    fprintf(stderr, "total_dyn_inst: %ld\n", dyn);
	  return;
	}
    }
  fprintf(stderr, "no main function found! exiting.\n");
}


value_t interp_fun(program_t *prog, size_t *dyn_insns,
		   size_t which_fun, value_t *args, size_t num_args)
{
  function_t *f = &prog->funcs[which_fun];
  value_t *context = malloc(sizeof(value_t) * f->num_tmps);
  uint16_t labels[] = {0, 0};
  memcpy(context, args, sizeof(value_t) * num_args);
  value_t tmp = interpret_insn(prog, which_fun, context, labels, dyn_insns, 0);
  free(context);
  return tmp;
}
