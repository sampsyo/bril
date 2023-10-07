#include "to_abstract_asm.h"

#define _POSIX_C_SOURCE 200809L
#include <string.h>

#include <stdio.h>

#ifdef __ARM_ARCH


static inline void make_space(tagged_arm_insn_t **insns, size_t *space, size_t idesired)
{
  while(idesired >= *space)
    {
      *space *= 2;
      *insns = realloc(*insns, *space * sizeof(asm_insn_t));
    }
}

static inline arm_arg_tagged_t from_tmp(uint16_t tmp)
{
  return (arm_arg_tagged_t) {.type = TMP, .value = (arm_arg_t) {.tmp = tmp}};
}

static inline arm_arg_tagged_t from_const(int16_t num)
{
  return (arm_arg_tagged_t) {.type = CNST, .value = (arm_arg_t) {.cnst = num}};
}

static inline void write_insn(asm_insn_t *insn, FILE *insn_stream)
{
  fwrite(insn, INSN_SIZE, 1, insn_stream);
}

static inline  void trans_const(FILE *insn_stream,
		 uint16_t dest, int64_t value)
{
  write_insn(&(tagged_arm_insn_t) {.type = AMOV,
			       .value = (arm_insn_t)
			       {.mov = (mov_arm_insn_t)
				{.dest = from_tmp(dest),
				 .src = from_const(value & 0xffff)}}},
    insn_stream);
  int64_t tmp = value >> 16;
  for(size_t i = 1; i  < 4; ++i)
    {
      if(tmp & 0xffff)
	write_insn(&(tagged_arm_insn_t) {.type = AMOVK,
					 .value = (arm_insn_t)
					 {.movk = (movk_arm_insn_t)
					  {.dest = from_tmp(dest),
					   .val = tmp & 0xffff,
					   .lsl = i * 16}}}, insn_stream);
      tmp >>= 16;
    }
}

norm_arm_op_t bril_op_to_arm(uint16_t op)
{
  switch(op)
    {
    case ADD: return AADD;
    case MUL: return AMUL;
    case SUB: return ASUB;
    case AND: return AAND;
    case  OR: return AORR;
    case DIV: return ASDIV;
    case FADD: return AFADD;
    case FMUL: return AFMUL;
    case FSUB: return AFSUB;
    case FDIV: return AFDIV;
    }
  return -1;
}

arm_cmp_flags_t bril_op_to_flag(uint16_t op)
{
  switch(op)
    {
    case EQ: return CMPEQ;
    case LT: return CMPLT;
    case LE: return CMPLE;
    case FEQ: return CMPEQ;
    case FLT: return CMPMI;
    case FLE: return CMPLS;
    case FGT:
    case GT: return CMPGT;
    case GE:
    case FGE: return CMPGE;
    }
  return -1;
}


asm_func_t trans_func(program_t *prog, size_t which_fun)
{
  char *mem_stream;
  size_t size_loc;
  //size_t insns_space = 32, ni = 0;
  //tagged_arm_insn_t *insns = malloc(sizeof(asm_insn_t) * insns_space);
  FILE *insn_stream = open_memstream(&mem_stream, &size_loc);
  function_t f = prog->funcs[which_fun];
  bool float_var = false;
  for(size_t i = 0; i < f.num_insns; ++i)
    {
      instruction_t *insn = f.insns + i;
      if(is_labelled(*insn))
	{
	  //make_space(&insns, &insns_space, ni);
	  arm_insn_t lbl;
	  sprintf(lbl.other, ".LF%s%lx:", f.name, i);
	  fwrite(&(tagged_arm_insn_t) {.type = AOTHER,
	    .value = lbl}, INSN_SIZE, 1, insn_stream);
	  /* insns[ni++] = (tagged_arm_insn_t) {.type = AOTHER, */
	  /*   .value = lbl}; */
	}
      uint16_t opcode = get_opcode(*insn);
      //make_space(&insns, &insns_space, ni);
      switch(opcode)
	{
	case NOP:
	  fwrite(&(tagged_arm_insn_t)
		 {.type = AOTHER,
		  .value = (arm_insn_t){ .other = "\tnop"}},
		 INSN_SIZE, 1, insn_stream);
	  break;
	case CONST:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = AMOVC, .value = (arm_insn_t)
		      {.movc = (movc_arm_insn_t)
		       {.dest = from_tmp(insn->const_insn.dest),
			.val = insn->const_insn.value}}}, insn_stream);
	  break;
	case LCONST:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = AMOVC, .value = (arm_insn_t)
		      {.movc = (movc_arm_insn_t)
		       {.dest = from_tmp(insn->long_const_insn.dest),
			.val = (insn + 1)->const_ext.int_val}}}, insn_stream);
	  ++i;
	  break;
	case ADD:
	case MUL:
	case SUB:
	case AND:
	case OR:
	case DIV:
	case FADD:
	case FMUL:
	case FSUB:
	case FDIV:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ANORM,
		      .value = (arm_insn_t)
		      {.norm = (norm_arm_insn_t)
		       {.op = bril_op_to_arm(opcode),
			.dest = from_tmp(insn->norm_insn.dest),
			.a1 = from_tmp(insn->norm_insn.arg1),
			.a2 = from_tmp(insn->norm_insn.arg2)}}
		     }, insn_stream);
	  break;
	case NOT:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ACMP,
		      .value = (arm_insn_t)
		      {.cmp = (cmp_arm_insn_t)
		       {.is_float = false,
			.a1 = from_tmp(insn->norm_insn.arg1),
			.a2 = from_const(0)
		       }}}, insn_stream);
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ASET, .value = (arm_insn_t)
		      {.set = (set_arm_insn_t)
		       {.dest = from_tmp(insn->norm_insn.dest),
			.flag = CMPEQ}}}, insn_stream);
	  break;
	case FEQ:
	case FLT:
	case FGT:
	case FGE:
	  float_var = true;
	  goto cmp_lbl;
	case EQ:
	case LT:
	case GT:
	case LE:
	case GE:
	  float_var = false;
	cmp_lbl:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ACMP,
		      .value = (arm_insn_t)
		      {.cmp = (cmp_arm_insn_t)
		       {.is_float = float_var,
			.a1 = from_tmp(insn->norm_insn.arg1),
			.a2 = from_tmp(insn->norm_insn.arg2)}}}, insn_stream);
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ASET, .value = (arm_insn_t)
		      {.set = (set_arm_insn_t)
		       {.dest = from_tmp(insn->norm_insn.dest),
			.flag = bril_op_to_flag(opcode)}}}, insn_stream);
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
		    write_insn(&(tagged_arm_insn_t)
			       {.type = AMOV, .value = (arm_insn_t)
				{.mov = (mov_arm_insn_t)
				 {.dest = (arm_arg_tagged_t)
				  {.type = REG, .value = (arm_arg_t)
				   {.reg = X0}},
				  .src = from_const(32)}}}, insn_stream);
		    write_insn(&(tagged_arm_insn_t)
		      {.type = ACALL, .value = (arm_insn_t)
		       {.call = (call_arm_insn_t)
			{.name = "putchar"}}}, insn_stream);
		  }
		switch(args[j])
		  {
		  case BRILINT:
		    write_insn(&(tagged_arm_insn_t)
			       {.type = AMOV, .value = (arm_insn_t)
				{.mov = (mov_arm_insn_t)
				 {.dest = (arm_arg_tagged_t)
				  {.type = REG, .value = (arm_arg_t)
				   {.reg = X0}},
				  .src = from_tmp(args[j + 1])}}}, insn_stream);
		    write_insn(&(tagged_arm_insn_t)
		      {.type = ACALL, .value = (arm_insn_t)
		       {.call = (call_arm_insn_t)
			{.name = "priint"}}}, insn_stream);
		    break;
		  case BRILBOOL:
		    write_insn(&(tagged_arm_insn_t)
			       {.type = AMOV, .value = (arm_insn_t)
				{.mov = (mov_arm_insn_t)
				 {.dest = (arm_arg_tagged_t)
				  {.type = REG, .value = (arm_arg_t)
				   {.reg = X0}},
				  .src = from_tmp(args[j + 1])}}}, insn_stream);
		    write_insn(&(tagged_arm_insn_t)
		      {.type = ACALL, .value = (arm_insn_t)
		       {.call = (call_arm_insn_t)
			{.name = "pribool"}}}, insn_stream);
		    break;
		  case BRILFLOAT:
		    write_insn(&(tagged_arm_insn_t)
			       {.type = AMOV, .value = (arm_insn_t)
				{.mov = (mov_arm_insn_t)
				 {.dest = (arm_arg_tagged_t)
				  {.type = REG, .value = (arm_arg_t)
				   {.reg = D0}},
				  .src = from_tmp(args[j + 1])}}}, insn_stream);
		    write_insn(&(tagged_arm_insn_t)
		      {.type = ACALL, .value = (arm_insn_t)
		       {.call = (call_arm_insn_t)
			{.name = "prifloat"}}}, insn_stream);
		  }
	      }
	    write_insn(&(tagged_arm_insn_t)
			       {.type = AMOV, .value = (arm_insn_t)
				{.mov = (mov_arm_insn_t)
				 {.dest = (arm_arg_tagged_t)
				  {.type = REG, .value = (arm_arg_t)
				   {.reg = X0}},
				  .src = from_const(10)}}}, insn_stream);
	    write_insn(&(tagged_arm_insn_t)
		       {.type = ACALL, .value = (arm_insn_t)
			{.call = (call_arm_insn_t)
			 {.name = "putchar"}}}, insn_stream);
	  }
	  break;
	case CALL:
	  {
	    uint16_t num_args = insn->call_inst.num_args;
	    uint16_t *args = (uint16_t*) (insn + 1);
	    function_t *target = prog->funcs + insn->call_inst.target;
	    tagged_arm_insn_t call = (tagged_arm_insn_t)
	      {.type = AABSCALL, .value = (arm_insn_t)
	       {.abs_call = (abs_call_arm_insn_t)
		{.num_args = num_args,
		 .ret_tp = target->ret_tp,
		 .dest = insn->call_inst.dest}}};
	    sprintf(call.value.abs_call.name, "%s", target->name);
	    write_insn(&call, insn_stream);
	    for(size_t eidx = 0; eidx < num_args; eidx += 32)
	      {
		abs_call_arm_ext_t ext;
		for(size_t argi = 0; eidx + argi < num_args && argi < 32; ++argi)
		  {
		    size_t arg = eidx + argi;
		    uint32_t tagged_arg = target->arg_types[arg];
		    tagged_arg <<= 16;
		    tagged_arg |= args[arg];
		    ext.typed_temps[argi] = tagged_arg;
		  }
		write_insn(&(tagged_arm_insn_t)
			   {.type = AABSEXT, .value = (arm_insn_t)
			    {.abs_call_ext = ext}}, insn_stream);
	      }
	    i += (num_args + 3) / 4;
	  }
	  break;
	case RET:
	  switch(f.ret_tp)
	    {
	    case BRILVOID:
	      break;
	    case BRILFLOAT:
	      write_insn(&(tagged_arm_insn_t)
			 {.type = AMOV, .value = (arm_insn_t)
			  {.mov = (mov_arm_insn_t)
			   {.dest = (arm_arg_tagged_t)
			    {.type = REG, .value = (arm_arg_t)
			     {.reg = D0}},
			    .src = from_tmp(insn->norm_insn.arg1),
			    .is_float = true}}}, insn_stream);
	      break;
	    default:
	      write_insn(&(tagged_arm_insn_t)
			 {.type = AMOV, .value = (arm_insn_t)
			  {.mov = (mov_arm_insn_t)
			   {.dest = (arm_arg_tagged_t)
			    {.type = REG, .value = (arm_arg_t)
			     {.reg = X0}},
			    .src = from_tmp(insn->norm_insn.arg1),
			    .is_float = false}}}, insn_stream);
	    }
	  arm_insn_t ins;
	  sprintf(ins.other, "\tb\t.L%s.ret", f.name);
	  write_insn(&(tagged_arm_insn_t)
		     {.type = AOTHER, .value = ins}, insn_stream);
	  break;
	case JMP:
	  {
	    arm_insn_t ins;
	    sprintf(ins.other, "\tb\t.LF%s%x", f.name, insn->norm_insn.dest);
	    write_insn(&(tagged_arm_insn_t)
		       {.type = AOTHER, .value = ins}, insn_stream);
	  }
	  break;
	case BR:
	  {
	    tagged_arm_insn_t ins = (tagged_arm_insn_t)
	      {.type = ACBZ, .value = (arm_insn_t)
	       {.cbz = (cbz_arm_insn_t)
		{.cond = from_tmp(insn->br_inst.test)}}};
	    sprintf(ins.value.cbz.dest, ".LF%s%x", f.name, insn->br_inst.lfalse);
	    write_insn(&ins, insn_stream);
	    ins = (tagged_arm_insn_t) {.type = AOTHER};
	    sprintf(ins.value.other, "\tb\t.LF%s%x", f.name, insn->br_inst.ltrue);
	    write_insn(&ins, insn_stream);
	  }
	  break;
	case ID:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = AMOV, .value = (arm_insn_t)
		      {.mov = (mov_arm_insn_t)
		       {.dest = from_tmp(insn->norm_insn.dest),
			.src = from_tmp(insn->norm_insn.arg1),
			.is_float = insn->norm_insn.arg2 == BRILFLOAT}}}, insn_stream);
	  break;
	case ALLOC:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ANORM, .value = (arm_insn_t)
		      {.norm = (norm_arm_insn_t)
		       {.op = ALSL,
			.dest = (arm_arg_tagged_t)
			{.type = REG, .value = (arm_arg_t)
			 {.reg = X0}},
		        .a1 = from_tmp(insn->norm_insn.arg1),
			.a2 = from_const(3)}}}, insn_stream);
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ACALL, .value = (arm_insn_t)
		      {.call = (call_arm_insn_t)
		       {.name = "malloc"}}}, insn_stream);
	  write_insn(&(tagged_arm_insn_t)
		     {.type = AMOV, .value = (arm_insn_t)
		      {.mov = (mov_arm_insn_t)
		       {.dest = from_tmp(insn->norm_insn.dest),
			.src = (arm_arg_tagged_t)
			{.type = REG, .value = (arm_arg_t)
			 {.reg = X0}},
			.is_float = false}}}, insn_stream);
	  break;
	case FREE:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = AMOV, .value = (arm_insn_t)
		      {.mov = (mov_arm_insn_t)
		       {.dest = (arm_arg_tagged_t)
			{.type = REG, .value = (arm_arg_t)
			 {.reg = X0}},
			.src = from_tmp(insn->norm_insn.arg1)}}}, insn_stream);
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ACALL, .value = (arm_insn_t)
		      {.call = (call_arm_insn_t)
		       {.name = "free"}}}, insn_stream);
	  break;
	case STORE:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ASTR, .value = (arm_insn_t)
		      {.str = (str_arm_insn_t)
		       {.address = from_tmp(insn->norm_insn.arg1),
			.value = from_tmp(insn->norm_insn.arg2)}}}, insn_stream);
	  break;
	case LOAD:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ALDR, .value = (arm_insn_t)
		      {.ldr = (ldr_arm_insn_t)
		       {.address = from_tmp(insn->norm_insn.arg1),
			.dest = from_tmp(insn->norm_insn.dest)}}}, insn_stream);
	  break;
	case PTRADD:
	  write_insn(&(tagged_arm_insn_t)
		     {.type = ANORM, .value = (arm_insn_t)
		      {.norm = (norm_arm_insn_t)
		       {.op = AADD,
			.dest = from_tmp(insn->norm_insn.dest),
			.a1 = from_tmp(insn->norm_insn.arg1),
			.a2 = from_tmp(insn->norm_insn.arg2),
			.lsl = 3}}}, insn_stream);
	  break;
	default:
	  fprintf(stderr, "unsupported opcode: %d\n", opcode);
	}
    }
  tagged_arm_insn_t i = (tagged_arm_insn_t)
    {.type = AOTHER, .value = (arm_insn_t) {}};
  sprintf(i.value.other, ".LF%s%lx:\n", f.name, f.num_insns);
  write_insn(&i, insn_stream);
  fclose(insn_stream);
  asm_func_t fun = (asm_func_t)
    {.num_insns = size_loc / sizeof(tagged_arm_insn_t),
     .num_temps = f.num_tmps,
     .num_args = f.num_args,
     .ret_tp = f.ret_tp,
     .insns = (asm_insn_t*) mem_stream};
  fun.arg_types = malloc(sizeof(uint16_t) * f.num_args);
  memcpy(fun.arg_types, f.arg_types, sizeof(uint16_t) * f.num_args);
  sprintf(fun.name, "%s", f.name);
  return fun;
}

asm_prog_t bytecode_to_abs_asm(program_t *prog)
{
  asm_func_t *funcs = malloc(sizeof(asm_func_t) * prog->num_funcs);
  for(size_t i = 0; i < prog->num_funcs; ++i)
    {
      funcs[i] = trans_func(prog, i);
    }
  asm_prog_t p;
  p.funcs = funcs;
  p.num_funcs = prog->num_funcs;
  return p;
}

#endif
