#include "trivial-regalloc.h"

#include "../bril-insns/types.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __ARM_ARCH

static inline arm_arg_tagged_t from_tmp(uint16_t tmp)
{
  return (arm_arg_tagged_t) {.type = TMP, .value = (arm_arg_t) {.tmp = tmp}};
}

static inline arm_arg_tagged_t from_reg(arm_reg_t r)
{
  return (arm_arg_tagged_t) {.type = REG, .value = (arm_arg_t) {.reg = r}};
}

static inline arm_arg_tagged_t from_const(int16_t num)
{
  return (arm_arg_tagged_t) {.type = CNST, .value = (arm_arg_t) {.cnst = num}};
}

static inline arm_arg_tagged_t from_uconst(uint16_t num)
{
  return (arm_arg_tagged_t) {.type = CNST, .value = (arm_arg_t) {.cnst = num}};
}


static inline int max(int a, int b)
{
  return a > b ? a : b;
}


static inline tagged_arm_insn_t mov(arm_reg_t dest, arm_arg_tagged_t src)
{
  switch(src.type)
    {
    case REG:
      return (tagged_arm_insn_t)
	{.type = AMOV, .value = (arm_insn_t)
	 {.mov = (mov_arm_insn_t)
	  {.dest = from_reg(dest), .src = src}}};
    case CNST:
      return (tagged_arm_insn_t)
	{.type = AMOV, .value = (arm_insn_t)
	 {.mov = (mov_arm_insn_t)
	  {.dest = from_reg(dest), .src = src}}};
    case TMP:
      return (tagged_arm_insn_t)
	{.type = ALDR, .value = (arm_insn_t)
	 {.ldr = (ldr_arm_insn_t)
	  {.dest = from_reg(dest),
	   .address = from_reg(SP),
	   .offset = 16 + src.value.tmp * 8}}};
    }
}

arm_arg_tagged_t float_arg_reg(int n)
{
  switch (n)
    {
    case 0: return from_reg(D0);
    case 1: return from_reg(D1);
    case 2: return from_reg(D2);
    case 3: return from_reg(D3);
    case 4: return from_reg(D4);
    case 5: return from_reg(D5);
    case 6: return from_reg(D6);
    case 7: return from_reg(D7);
    }
  fprintf(stderr, "invalid float arg %d\n", n);
  exit(1);
}


arm_arg_tagged_t norm_arg_reg(int n)
{
  switch (n)
    {
    case 0: return from_reg(X0);
    case 1: return from_reg(X1);
    case 2: return from_reg(X2);
    case 3: return from_reg(X3);
    case 4: return from_reg(X4);
    case 5: return from_reg(X5);
    case 6: return from_reg(X6);
    case 7: return from_reg(X7);
    }
  fprintf(stderr, "invalid regular arg %d\n", n);
  //exit(1);
}

static inline tagged_arm_insn_t movd(arm_arg_tagged_t dest, arm_reg_t src)
{
  switch(dest.type)
    {
    case REG:
      return (tagged_arm_insn_t)
	{.type = AMOV, .value = (arm_insn_t)
	 {.mov = (mov_arm_insn_t)
	  {.dest = dest, .src = from_reg(src)}}};
    case TMP:
      return (tagged_arm_insn_t)
	{.type = ASTR, .value = (arm_insn_t)
	 {.str = (str_arm_insn_t)
	  {.value = from_reg(src),
	   .address = from_reg(SP),
	   .offset = 16 + dest.value.tmp * 8}}};
    default:
      fprintf(stderr, "cannot move into a constant\n");
      exit(1);
    }
}

void write_insn(asm_insn_t insn, FILE *insn_stream)
{
  fwrite(&insn, INSN_SIZE, 1, insn_stream);
}

static inline bool reg_eql(arm_reg_t reg, arm_arg_tagged_t arg)
{
  return arg.type == REG && arg.value.reg == reg;
}

bool is_float(norm_arm_op_t op)
{
  switch (op)
    {
    case AFADD:
    case AFMUL:
    case AFSUB:
    case AFDIV:
      return true;
    default:
      return false;
    }
}

void trans_const(FILE *insn_stream,
		arm_arg_tagged_t dest, int64_t value)
{
  write_insn((tagged_arm_insn_t) {.type = AMOV,
				  .value = (arm_insn_t)
				  {.mov = (mov_arm_insn_t)
				   {.dest = from_reg(X0),
				    .src = from_uconst(value & 0xffff)}}},
    insn_stream);
  int64_t tmp = value >> 16;
  for(size_t i = 1; i  < 4; ++i)
    {
      if(tmp & 0xffff)
	write_insn((tagged_arm_insn_t) {.type = AMOVK,
					.value = (arm_insn_t)
					{.movk = (movk_arm_insn_t)
					 {.dest = from_reg(X0),
					  .val = tmp & 0xffff,
					  .lsl = i * 16}}}, insn_stream);
      tmp >>= 16;
    }
  write_insn(movd(dest, X0), insn_stream);
}


size_t trivial_prologue(asm_func_t f, FILE *insn_stream)
{
  size_t stack_offset = 8 * f.num_temps + 8;
  bool is_main = strcmp(f.name, "main") == 0;
  if(is_main && f.num_args != 0)
    stack_offset += 8;
  if(stack_offset % 16 != 0)
    stack_offset += 8;
  if(stack_offset < 512)
    {
      tagged_arm_insn_t i = (tagged_arm_insn_t)
	{.type = AOTHER, .value = (arm_insn_t) {}};
      sprintf(i.value.other, "\tstp\tx29, x30, [sp, -%ld]!", stack_offset);
      write_insn(i, insn_stream);
    } else
    {
      write_insn((tagged_arm_insn_t)
		 {.type = ANORM, .value = (arm_insn_t)
		  {.norm = (norm_arm_insn_t)
		   {.op = ASUB, .dest = from_reg(SP),
		    .a1 = from_reg(SP), .a2 = from_const(stack_offset)}}}, insn_stream);
      tagged_arm_insn_t i = (tagged_arm_insn_t)
	{.type = AOTHER, .value  = (arm_insn_t) {}};
      sprintf(i.value.other, "\tstp\tx29, x30, [sp]");
      write_insn(i, insn_stream);
    }
  write_insn(mov(X29, from_reg(SP)), insn_stream);
  if(is_main && f.num_args != 0)
    {
      write_insn((tagged_arm_insn_t)
		 {.type = ASTR, .value = (arm_insn_t)
		  {.str = (str_arm_insn_t)
		   {.value = from_reg(X19),
		    .address = from_reg(SP),
		    .offset = stack_offset - 16}}}, insn_stream);
      write_insn(mov(X19, from_reg(X1)), insn_stream);
      for(size_t i = 0; i < f.num_args; ++i)
	{
	  write_insn((tagged_arm_insn_t)
		     {.type = ALDR, .value = (arm_insn_t)
		      {.ldr = (ldr_arm_insn_t)
		       {.dest = from_reg(X0),
			.address = from_reg(X19),
			.offset = (i + 1) * 8}}}, insn_stream);
	  switch(f.arg_types[i])
	    {
	    case BRILINT:
	      write_insn((tagged_arm_insn_t)
			 {.type = ACALL, .value = (arm_insn_t)
			  {.call = (call_arm_insn_t)
			   {.name = "int_of_string"}}}, insn_stream);
	      write_insn((tagged_arm_insn_t)
			 {.type = ASTR, .value = (arm_insn_t)
			  {.str = (str_arm_insn_t)
			   {.value = from_reg(X0),
			    .address = from_reg(SP),
			    .offset = 16 + i * 8}}}, insn_stream);
	      break;
	    case BRILBOOL:
	      write_insn((tagged_arm_insn_t)
			 {.type = ACALL, .value = (arm_insn_t)
			  {.call = (call_arm_insn_t)
			   {.name = "bool_of_string"}}}, insn_stream);
	      write_insn((tagged_arm_insn_t)
			 {.type = ASTR, .value = (arm_insn_t)
			  {.str = (str_arm_insn_t)
			   {.value = from_reg(X0),
			    .address = from_reg(SP),
			    .offset = 16 + i * 8}}}, insn_stream);
	      break;
	    case BRILFLOAT:
	      write_insn((tagged_arm_insn_t)
			 {.type = ACALL, .value = (arm_insn_t)
			  {.call = (call_arm_insn_t)
			   {.name = "float_of_string"}}}, insn_stream);
	      write_insn((tagged_arm_insn_t)
			 {.type = ASTR, .value = (arm_insn_t)
			  {.str = (str_arm_insn_t)
			   {.value = from_reg(D0),
			    .address = from_reg(SP),
			    .offset = 16 + i * 8}}}, insn_stream);
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
		  write_insn((tagged_arm_insn_t)
			     {.type = ALDR, .value = (arm_insn_t)
			      {.ldr = (ldr_arm_insn_t)
			       {.dest = from_reg(D9),
				.address = from_reg(SP),
				.offset = stack_offset + 8 * spilled_args++}}},
			     insn_stream);
		  write_insn(movd(from_tmp(++double_args), D9), insn_stream);
		}
	    } else
	    {
	      if(other_args < 7)
		++other_args;
	      else
		{
		   write_insn((tagged_arm_insn_t)
			     {.type = ALDR, .value = (arm_insn_t)
			      {.ldr = (ldr_arm_insn_t)
			       {.dest = from_reg(X9),
				.address = from_reg(SP),
				.offset = stack_offset + 8 * spilled_args++}}},
			     insn_stream);
		   write_insn(movd(from_tmp(++other_args), X9), insn_stream);
		}
	    }
	}
      double_args = 0; other_args = 0;
      for(size_t i = 0; i < f.num_args; ++i)
	{
	  if(f.arg_types[i] == BRILFLOAT)
	    {
	      if(double_args < 8)
		write_insn(movd(from_tmp(i), float_arg_reg(double_args++).value.reg),
			   insn_stream);
	    } else
	    {
	      if(other_args < 8)
		write_insn(movd(from_tmp(i), norm_arg_reg(other_args++).value.reg),
			   insn_stream);
	    }
	}
    }
  return stack_offset;
}


void trivial_epilogue(asm_func_t f, FILE *insn_stream, size_t stack_offset)
{
  tagged_arm_insn_t i = (tagged_arm_insn_t)
    {.type = AOTHER, .value = (arm_insn_t) {}};
  sprintf(i.value.other, ".L%s.ret:", f.name);
  write_insn(i, insn_stream);
  bool is_main = strcmp(f.name, "main") == 0;
  if(is_main && f.num_args != 0)
    write_insn((tagged_arm_insn_t)
	       {.type = ALDR, .value = (arm_insn_t)
		{.ldr = (ldr_arm_insn_t)
		 {.dest = from_reg(X19),
		  .address = from_reg(SP),
		  .offset = stack_offset - 16}}}, insn_stream);
  if(is_main && f.ret_tp == BRILVOID)
    write_insn((tagged_arm_insn_t)
	       {.type = AOTHER, .value = (arm_insn_t)
		{.other = "\tmov\tw0, 0"}}, insn_stream);
  if(stack_offset < 512)
    {
      tagged_arm_insn_t i = (tagged_arm_insn_t)
	{.type = AOTHER, .value = (arm_insn_t) {}};
      sprintf(i.value.other, "\tldp\tx29, x30, [sp], %ld\n", stack_offset);
      write_insn(i, insn_stream);
    } else
    {
      write_insn((tagged_arm_insn_t)
		 {.type = AOTHER, .value = (arm_insn_t)
		  {.other = "\tldp\tx29, x30, [sp]"}}, insn_stream);
      write_insn((tagged_arm_insn_t)
		 {.type = ANORM, .value = (arm_insn_t)
		  {.norm = (norm_arm_insn_t)
		   {.op = AADD, .dest = from_reg(SP),
		    .a1 = from_reg(SP), .a2 = from_const(stack_offset)}}}, insn_stream);
    }
  write_insn((tagged_arm_insn_t)
	     {.type = AOTHER, .value = (arm_insn_t)
	      {.other = "ret"}}, insn_stream);
}

asm_func_t allocate(asm_prog_t *p, size_t which_fun)
{
  char *mem_stream;
  size_t size_loc;
  FILE *insn_stream = open_memstream(&mem_stream, &size_loc);
  asm_func_t f = p->funcs[which_fun];
  size_t stack_offset = trivial_prologue(f, insn_stream);
  for(size_t i = 0; i < f.num_insns; ++i)
    {
      tagged_arm_insn_t insn = f.insns[i];
      switch(insn.type)
	{
	case ANORM:
	  {
	    arm_reg_t r1 = is_float(insn.value.norm.op) ? D0 : X0;
	    arm_reg_t r2 = is_float(insn.value.norm.op) ? D1 : X1;
	    bool is_weird = reg_eql(r1, insn.value.norm.a2);
	    if(is_weird)
	      write_insn(mov(r2, insn.value.norm.a2), insn_stream);
	    write_insn(mov(r1, insn.value.norm.a1), insn_stream);
	    if(!is_weird)
	      write_insn(mov(r2, insn.value.norm.a2), insn_stream);
	    write_insn((tagged_arm_insn_t)
		       {.type = ANORM, .value = (arm_insn_t)
			{.norm = (norm_arm_insn_t)
			 {.op = insn.value.norm.op,
			  .a1 = from_reg(r1),
			  .a2 = from_reg(r2),
			  .dest = from_reg(r1),
			  .lsl = insn.value.norm.lsl}}}, insn_stream);
	    write_insn(movd(insn.value.norm.dest, r1), insn_stream);
	  } break;
	case ACMP:
	  {
	    bool is_float = insn.value.cmp.is_float;
	    arm_reg_t r1 = is_float ? D0 : X0;
	    arm_reg_t r2 = is_float ? D1 : X1;
	    bool is_weird = reg_eql(r1, insn.value.cmp.a2);
	    if(is_weird)
	      write_insn(mov(r2, insn.value.cmp.a2), insn_stream);
	    write_insn(mov(r1, insn.value.cmp.a1), insn_stream);
	    if(!is_weird)
	      write_insn(mov(r2, insn.value.cmp.a2), insn_stream);
	    write_insn((tagged_arm_insn_t)
		       {.type = ACMP, .value = (arm_insn_t)
			{.cmp = (cmp_arm_insn_t)
			 {.is_float = is_float,
			  .a1 = from_reg(r1),
			  .a2 = from_reg(r2)}}}, insn_stream);
	  } break;
	case ASET:
	  write_insn((tagged_arm_insn_t)
		     {.type = ASET, .value = (arm_insn_t)
		      {.set = (set_arm_insn_t)
		       {.dest = from_reg(X0),
			.flag = insn.value.set.flag}}}, insn_stream);
	  write_insn(movd(insn.value.set.dest, X0), insn_stream);
	  break;
	case AMOV:
	  if(insn.value.mov.dest.type == REG)
	    {
	      write_insn(mov(insn.value.mov.dest.value.reg,
			     insn.value.mov.src), insn_stream);
	    } else
	    {
	      arm_reg_t reg = insn.value.mov.is_float ? D0 : X0;
	      write_insn(mov(reg, insn.value.mov.src), insn_stream);
	      write_insn(movd(insn.value.mov.dest, reg), insn_stream);
	    }
	  break;
	case AMOVC:
	  trans_const(insn_stream, insn.value.movc.dest, insn.value.movc.val);
	  break;
	case ACBZ:
	  write_insn(mov(X0, insn.value.cbz.cond), insn_stream);
	  tagged_arm_insn_t in = (tagged_arm_insn_t)
	    {.type = ACBZ, .value = (arm_insn_t)
	     {.cbz = (cbz_arm_insn_t)
	      {.cond = from_reg(X0)}}};
	  sprintf(in.value.cbz.dest, "%s", insn.value.cbz.dest);
	  write_insn(in, insn_stream);
	  break;
	case AABSCALL:
	  {
	    uint16_t num_args = insn.value.abs_call.num_args;
	    uint32_t *typed_args = alloca(sizeof(uint32_t) * num_args);
	    for(size_t x = 0; x < num_args; x += 32)
	      {
		for(size_t argi = 0; x * 32 + argi < num_args && argi < 32; ++argi)
		  typed_args[x * 32 + argi] = f.insns[i + 1 + x/32]
		    .value.abs_call_ext.typed_temps[argi];
		++i;
	      }
	    uint16_t norm_args = 0, float_args = 0;
	    for(uint16_t i = 0; i < num_args; ++i)
	      {
		if(typed_args[i] >> 16 == BRILFLOAT)
		  ++float_args;
		else
		  ++norm_args;
	      }
	    int norm_stack_needed = max((norm_args - 9) * 8, 0);
	    int float_stack_needed = max((float_args - 8) * 8, 0);
	    int stack_needed = norm_stack_needed + float_stack_needed;
	    if(stack_needed % 16 != 0) stack_needed += 8;
	    if(stack_needed)
	      write_insn((tagged_arm_insn_t)
			 {.type = ANORM, .value = (arm_insn_t)
			  {.norm = (norm_arm_insn_t)
			   {.op = ASUB, .dest = from_reg(SP),
			    .a1 = from_reg(SP), .a2 = from_const(stack_needed)}}},
			 insn_stream);
	    int norm_args_left = norm_args;
	    int float_args_left = float_args;
	    int stack_args_left = max(norm_args - 8, 0) + max(float_args - 8, 0) - 1;
	    for(int argidx = num_args - 1; argidx >= 0; --argidx)
	      {
		int *left;
		arm_arg_tagged_t (*get_arg)(int);
		if(typed_args[argidx] >> 16 == BRILFLOAT)
		  {
		    left = &float_args_left;
		    get_arg = &float_arg_reg;
		  } else
		  {
		    left = &norm_args_left;
		    get_arg = &norm_arg_reg;
		  }
		if(*left > 8)
		  {
		    write_insn((tagged_arm_insn_t)
			       {.type = ALDR, .value = (arm_insn_t)
				{.ldr = (ldr_arm_insn_t)
				 {.dest = from_reg(X0),
				  .address = from_reg(SP),
				  .offset = 16 + 8 * (typed_args[argidx] & 0xffff)
				  + stack_needed}}}, insn_stream);
		    write_insn((tagged_arm_insn_t)
			       {.type = ASTR, .value = (arm_insn_t)
				{.str = (str_arm_insn_t)
				 {.value = from_reg(X0),
				  .address = from_reg(SP),
				  .offset = stack_args_left * 8}}}, insn_stream);
		  } else
		  {
		    write_insn((tagged_arm_insn_t)
			       {.type = ALDR, .value = (arm_insn_t)
				{.ldr = (ldr_arm_insn_t)
				 {.dest = get_arg(*left - 1),
				  .address = from_reg(SP),
				  .offset = 16 + 8 * (typed_args[argidx] & 0xffff)
				  + stack_needed}}}, insn_stream);
		  }
		--(*left);
		--stack_args_left;
	      }
	    tagged_arm_insn_t call_insn = (tagged_arm_insn_t)
	      {.type = ACALL};
	    sprintf(call_insn.value.call.name, "%s", insn.value.abs_call.name);
	    write_insn(call_insn, insn_stream);
	    if(stack_needed)
	      write_insn((tagged_arm_insn_t)
			 {.type = ANORM, .value = (arm_insn_t)
			  {.norm = (norm_arm_insn_t)
			   {.op = AADD, .dest = from_reg(SP),
			    .a1 = from_reg(SP), .a2 = from_const(stack_needed)}}},
			 insn_stream);
	    if(insn.value.abs_call.ret_tp == BRILFLOAT)
	      write_insn(movd(from_tmp(insn.value.abs_call.dest), D0), insn_stream);
	    else if(insn.value.abs_call.ret_tp != BRILVOID)
	      write_insn(movd(from_tmp(insn.value.abs_call.dest), X0), insn_stream);
	  } break;
	case ALDR:
	  write_insn(mov(X0, insn.value.ldr.address), insn_stream);
	  write_insn((tagged_arm_insn_t)
		     {.type = ALDR, .value = (arm_insn_t)
		      {.ldr = (ldr_arm_insn_t)
		       {.dest = from_reg(X0),
			.address = from_reg(X0),
			.offset = insn.value.ldr.offset}}}, insn_stream);
	  write_insn(movd(insn.value.ldr.dest, X0), insn_stream);
	  break;
	case ASTR:
	  write_insn(mov(X0, insn.value.str.value), insn_stream);
	  write_insn(mov(X1, insn.value.str.address), insn_stream);
	  write_insn((tagged_arm_insn_t)
		     {.type = ASTR, .value = (arm_insn_t)
		      {.str = (str_arm_insn_t)
		       {.value = from_reg(X0),
			.address = from_reg(X1),
			.offset = insn.value.str.offset}}}, insn_stream);
	  break;
	case AMOVK:
	case AOTHER:
	case ACALL:
	  write_insn(insn, insn_stream);
	  break;
	default:
	  fprintf(stderr, "bad insn type: %d\n", insn.type);
	  exit(1);
	}
    }
  trivial_epilogue(f, insn_stream, stack_offset);
  fclose(insn_stream);
  asm_func_t ret;
  sprintf(ret.name, "%s", f.name);
  ret.insns = (asm_insn_t*) mem_stream;
  ret.num_insns = size_loc / sizeof(tagged_arm_insn_t);
  /* other metadata no longer needed since not abstract*/
  return ret;
}

asm_prog_t triv_allocate(asm_prog_t p)
{
  asm_func_t *funs = malloc(sizeof(asm_func_t) * p.num_funcs);

  for(size_t i = 0; i < p.num_funcs; ++i)
    {
	funs[i] = allocate(&p, i);
    }
  return (asm_prog_t)
    {.funcs = funs,
     .num_funcs = p.num_funcs};
}

#else

asm_prog_t triv_allocate(asm_prog_t p)
{
  fprintf(stderr, "arch not supported");
  exit(1);
}

#endif
