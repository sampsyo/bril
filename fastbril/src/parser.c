#include "parser.h"
#include "libs/hashmap.h"
#include <stdio.h>

#define TEST_OP(s, ret) if(strcmp(s, str) == 0) { return (ret);}
#define MAKE_HASH_MAP hashmap_new(sizeof(struct string_uint16), 0, 0, 0, \
				  hashfun, hash_compare, NULL, NULL);


/**
 * translate strings into our internal opcode number
 */
static inline uint16_t opcode_of_string(const char *str)
{
  TEST_OP( "nop",    NOP);
  TEST_OP( "const",  CONST);
  TEST_OP( "add",    ADD);
  TEST_OP( "mul",    MUL);
  TEST_OP( "mul",    MUL);
  TEST_OP( "sub",    SUB);
  TEST_OP( "div",    DIV);
  TEST_OP( "eq",     EQ);
  TEST_OP( "lt",     LT);
  TEST_OP( "gt",     GT);
  TEST_OP( "le",     LE);
  TEST_OP( "ge",     GE);
  TEST_OP( "not",    NOT);
  TEST_OP( "and",    AND);
  TEST_OP( "or",     OR);
  TEST_OP( "jmp",    JMP);
  TEST_OP( "br",     BR);
  TEST_OP( "call",   CALL);
  TEST_OP( "ret",    RET);
  TEST_OP( "print",  PRINT);
  TEST_OP( "phi",    PHI);
  TEST_OP( "alloc",  ALLOC);
  TEST_OP( "free",   FREE);
  TEST_OP( "store",  STORE);
  TEST_OP( "load",   LOAD);
  TEST_OP( "ptradd", PTRADD);
  TEST_OP( "fadd",   FADD);
  TEST_OP( "fmul",   FMUL);
  TEST_OP( "fsub",   FSUB);
  TEST_OP( "fdiv",   FDIV);
  TEST_OP( "feq",    FEQ);
  TEST_OP( "flt",    FLT);
  TEST_OP( "fle",    FLE);
  TEST_OP( "fgt",    FGT);
  TEST_OP( "fge",    FGE);
  TEST_OP( "id",     ID);
  return 0xffff;
}

static inline uint16_t type_of_string(const char *str)
{
  TEST_OP( "int",    BRILINT);
  TEST_OP( "bool",   BRILBOOL);
  TEST_OP( "float",  BRILFLOAT);
  return 0xffff;
}

/**
 * take value, which is in json form, and convert it to a type.
 * undefined behaviour if not actually a type
 */
static inline uint16_t type_of_json_value(struct json_value_s *value)
{
  if(value == 0)
    return BRILVOID;
  uint16_t ptr_depth = 0;
  while(value->type == json_type_object)
    {
      if(ptr_depth >= 1 << 14)
	{
	  fprintf(stderr, "pointers nested too deep\n");
	  exit(1);
	}
      ++ptr_depth;
      value = json_value_as_object(value)->start->value;
    }
  uint16_t base_tp = type_of_string(json_value_as_string(value)->string);
  return ptr_depth << 2 | base_tp;
}

/**
 * boilerplate to make the hashmap work properly. No need to understand this.
 */
typedef struct string_uint16
{
  const char *str;
  uint16_t num;
} hashdat;

uint64_t hashfun(const void *item, uint64_t seed0, uint64_t seed1) {
  const struct string_uint16 *val = item;
  return hashmap_sip(val->str, strlen(val->str), seed0, seed1);
}

int hash_compare(const void *a, const void *b, void *udata) {
  const struct string_uint16 *ua = a;
  const struct string_uint16 *ub = b;
  return strcmp(ua->str, ub->str);
}

/**
 * Parse a temp value (variable) from the json tmp.
 * tmp_map is the current mapping from names -> numbers, as internally we
 * represent temps as numbers. If tmp is already mapped, we will return that,
 * but if it isn't we will map it to *num_tmps, and then increment this
 * variable to reflect the new number of mapped temps.
 */
static inline uint16_t parse_temp(struct json_value_s *tmp,
		    struct hashmap *tmp_map,
		    uint16_t *num_tmps)
{
  const char *nm = json_value_as_string(tmp)->string;
  hashdat *precomped = hashmap_get(tmp_map, &(hashdat){.str = nm});
  if (precomped)
    {
      //printf("found %s -> %d\n", nm, precomped->num);
      return precomped->num;
    } else
    {
      if(*num_tmps + 1 == 0xffff)
	{
	  fprintf(stderr, "too many variables!!!\n");
	  exit(1);
	}
      uint16_t tmp  = *num_tmps;
      *num_tmps = *num_tmps + 1;
      hashmap_set(tmp_map, &(hashdat){.str = nm, .num = tmp});
      return tmp;
    }
}


/**
 * take a label in the 1st representation (indexed by order seen) and
 * turn it into the final representation (instruction index).
 * see below for details.
 */
static inline uint16_t translate_label(struct hashmap *lbl_map,
			 const char **idx_to_lbl, uint16_t old_lbl)
{
  return ((hashdat*) hashmap_get(lbl_map, &(hashdat)
				 {.str = idx_to_lbl[old_lbl]}))->num;
}

/**
 * parse the json value lbl as a label. Update appropriate state as necessary.
 */
static inline uint16_t parse_lbls(struct json_value_s *lbl,
		    struct hashmap *prt_lbl_map,
		    const char **idx_to_lbl,
		    uint16_t *num_lbls)
{
  const char *nm = json_value_as_string(lbl)->string;
  hashdat *precomped = hashmap_get(prt_lbl_map, &(hashdat){.str = nm});
  if (precomped)
    {
      return precomped->num;
    } else
    {
      uint16_t new_lbl = (*num_lbls)++;
      hashmap_set(prt_lbl_map, &(hashdat){.str = nm, .num = new_lbl});
      idx_to_lbl[new_lbl] = nm;
      return new_lbl;
    }
}


/**
 * Parse a single instruction out of json.
 * put this parsed instruction into the array insns[dest], resizing as necessary
 * insn_length is the length of the array insns.
 * next_labelled is whether the next instruction is tagged by a label
 *                                                    (needed for ssa)
 * lbl_map is a map from label names to their indices in insns
 * tmp_map is a map from temp names to their number representations
 * prt_lbl_map is a map from label names to numbers which represent the order
 *  in which labels were encountered. We use this since when we encounter a jump
 *  to a label, we can't know the actual representation of this label until
 *  later, so we need to put in a filler value to be replaced later
 * idx_to_lbl is the inverse of prt_lbl_map
 * num_lbls, num_tmps are what they sound like
 * tmp_types is a mapping from the internal rep of tmps to their types
 */
size_t parse_instruction(struct json_object_s *json,
			 instruction_t **insns,
			 size_t dest,
			 size_t *insn_length,
			 uint16_t *next_labelled,
			 struct hashmap *lbl_map,
			 struct hashmap *tmp_map,
			 struct hashmap *prt_lbl_map,
			 struct hashmap *fun_name_to_idx,
			 const char **idx_to_lbl,
			 uint16_t *num_lbls,
			 uint16_t *num_tmps,
			 uint16_t *tmp_types
			 )
/* when see label: string -> instruction index */
/* when see jmp lbl: prt_lbl_map[lbl] || {prt_lbl_map[lbl] = *num_lbls;*/
/*                                        idx_to_lbl[*num_lbls] = lbl;*/
/*                                        ++*num_lbls;}*/
/* when see jmp:   (string, ordered index) */
{

  struct json_object_element_s *field = json->start;
  uint16_t tagged_opcode = 0xffff;
  uint16_t opcode = 0;
  uint16_t insn_dest = 0xffff;
  bool is_label = false;
  uint16_t *args = 0;

  size_t numargs = 0;
  size_t num_cur_lbls = 0;
  uint16_t *lbls = 0;
  uint16_t type = 0xffff;
  const char *value = 0;
  const char *fun_nm = 0;

  if(dest + 1 == 0xffff)
    {
      fprintf(stderr, "too many instructions. please modularize your code\n");
      exit(1);
    }
  while(field)
    {
      if(strcmp(field->name->string, "op") == 0)
	{
	  opcode = opcode_of_string(json_value_as_string(field->value)->string);
	  tagged_opcode = opcode | (*next_labelled ? 0x8000 : 0);
	} else if (strcmp(field->name->string, "label") == 0)
	{
	  is_label = true;
	  const char *nm = json_value_as_string(field->value)->string;
	  hashmap_set(lbl_map, &(hashdat){.str = nm, .num = dest});
	} else if (strcmp(field->name->string, "dest") == 0)
	{
	  insn_dest = parse_temp(field->value, tmp_map, num_tmps);
	} else if (strcmp(field->name->string, "args") == 0)
	{
	  struct json_array_s *arr = field->value->payload;
	  numargs = arr->length;
	  args = malloc(sizeof(uint16_t) * numargs);
	  struct json_array_element_s *elem = arr->start;
	  for(int i = 0; i < numargs; ++i)
	    {
	      args[i] = parse_temp(elem->value, tmp_map, num_tmps);
	      elem = elem->next;
	    }
	} else if (strcmp(field->name->string, "labels") == 0)
	{
	  struct json_array_s *arr = field->value->payload;
	  num_cur_lbls = arr->length;
	  lbls = malloc(sizeof(uint16_t) * num_cur_lbls);
	  struct json_array_element_s *elem = arr->start;
	  for(int i = 0; i < num_cur_lbls; ++i)
	    {
	      lbls[i] = parse_lbls(elem->value, prt_lbl_map, idx_to_lbl, num_lbls);
	      elem = elem->next;
	    }
	} else if (strcmp(field->name->string, "type") == 0)
	{
	  type = type_of_json_value(field->value);
	} else if (strcmp(field->name->string, "value") == 0)
	{
	  switch(field->value->type)
	    {
	    case json_type_true:
	      value = "1";
	      break;
	    case json_type_false:
	      value = "0";
	      break;
	    case json_type_number:
	      value = json_value_as_number(field->value)->number;
	      break;
	    default:
	      fprintf(stderr, "bad constant. exiting\n");
	      exit(1);
	    }
	} else if (strcmp(field->name->string, "funcs") == 0)
	{
	  fun_nm = json_value_as_string
	    (json_value_as_array(field->value)->start->value)->string;
	}
      field = field->next;
    }
  /* add this variable to the type map (maybe) */
  if(type != 0xffff)
    {
      tmp_types[insn_dest] = type;
    }
  size_t extra_words_needed = 0;


  /* calculate multi-word instructions*/
  if(opcode == PHI)
    {
      extra_words_needed = (numargs + 1)/2;
    } else if (opcode == PRINT)
    {
      extra_words_needed = numargs / 2;
    } else if (opcode == CALL)
    {
      extra_words_needed = ((numargs + 3) / 4);
    } else if (opcode == CONST && (type == BRILINT || type == BRILBOOL))
    {
      int64_t val = strtoll(value, 0, 0);
      if ((int64_t) ((int32_t) val) != val)
	goto set_long_const;
    } else if (opcode == CONST && type == BRILFLOAT)
    {
    set_long_const:
      opcode = LCONST;
      tagged_opcode = (tagged_opcode & 0x8000 ? opcode | 0x8000 : opcode);
      extra_words_needed = 1;
    }
  /* realloc when we run out of space */
  if(dest + extra_words_needed >= *insn_length)
    {
      (*insn_length) *= 2;
      *insns = realloc(*insns, *insn_length * sizeof(instruction_t));
    }

  /* do actual emission */
  switch (opcode)
    {
    case 0: break;
    case PHI:
      {
	(*insns)[dest].phi_inst = (phi_inst_t)
	  {
	    .opcode_lbled = tagged_opcode,
	    .dest = insn_dest,
	    .num_choices = numargs
	  };
	for(size_t phi_ext_idx = 0; phi_ext_idx < numargs; phi_ext_idx += 2)
	  {
	    phi_extension_t ext;
	    ext.lbl1 = lbls[phi_ext_idx];
	    ext.val1 = args[phi_ext_idx];
	    if(phi_ext_idx + 1 < numargs)
	      {
		ext.lbl2 = lbls[phi_ext_idx + 1];
		ext.val2 = args[phi_ext_idx + 1];
	      }
	    (*insns)[dest + (phi_ext_idx/2) + 1].phi_ext = ext;
	  }
      } break;
    case PRINT:
      {
	(*insns)[dest].print_insn = (print_instr_t)
	  {
	    .opcode_lbled = tagged_opcode,
	    .num_prints = numargs,
	    .arg1 = args[0]
	  };
	for(size_t xtra_arg = 1; xtra_arg < numargs; xtra_arg += 2)
	  {
	    print_args_t pa;
	    pa.arg1 = args[xtra_arg];
	    if(xtra_arg + 1 < numargs)
	      pa.arg2 = args[xtra_arg + 1];
	    (*insns)[dest + (xtra_arg - 1)/2 + 1].print_args = pa;
	  }
      } break;
    case CALL:
      {
	(*insns)[dest].call_inst = (call_inst_t)
	  {
	    .opcode_lbled = tagged_opcode,
	    .dest = insn_dest,
	    .num_args = numargs,
	    .target = ((hashdat*) hashmap_get(fun_name_to_idx,
					      &(hashdat){.str = fun_nm}))->num,
	  };
	for(size_t arg = 0; arg < numargs; arg += 4)
	  {
	    call_args_t ca;
	    ca.args[0] = args[arg];
	    if(arg + 1 < numargs)
	      ca.args[1] = args[arg + 1];
	    if(arg + 2 < numargs)
	      ca.args[2] = args[arg + 2];
	    if(arg + 3 < numargs)
	      ca.args[3] = args[arg + 3];
	    (*insns)[dest + arg/4 + 1].call_args = ca;
	  }
      } break;
    case LCONST:
      {
	(*insns)[dest].long_const_insn = (long_const_instr_t)
	  {.opcode_lbled = tagged_opcode,
	   .dest = insn_dest,
	   .type = type};
	if(type == BRILFLOAT)
	  (*insns)[dest + 1].const_ext.float_val = strtod(value, 0);
	else
	  (*insns)[dest + 1].const_ext.int_val = strtoll(value, 0, 0);
      } break;
    case CONST:
      {
	(*insns)[dest].const_insn = (const_instr_t)
	  {
	    .opcode_lbled = tagged_opcode,
	    .dest = insn_dest,
	    .value = (int32_t) strtol(value, 0, 0)
	  };
      } break;
    case BR:
      {
	(*insns)[dest].br_inst = (br_inst_t)
	  {
	    .opcode_lbled = tagged_opcode,
	    .test = args[0],
	    .ltrue = lbls[0],
	    .lfalse = lbls[1]
	  };
      } break;
    case JMP:
      {
	(*insns)[dest].norm_insn = (norm_instruction_t)
	  {
	    .opcode_lbled = tagged_opcode,
	    .dest = lbls[0],
	    .arg1 = 0,
	    .arg2 = 0
	  };
      } break;
    case ID:
      {
	(*insns)[dest].norm_insn = (norm_instruction_t)
	  {
	    .opcode_lbled = tagged_opcode,
	    .dest = insn_dest,
	    .arg1 = args[0],
	    .arg2 = type
	  };
      } break;
    default:
      {
	(*insns)[dest].norm_insn = (norm_instruction_t)
	  {
	    .opcode_lbled = tagged_opcode,
	    .dest = insn_dest,
	    .arg1 = numargs > 0 ? args[0] : 0xffff,
	    .arg2 = numargs > 1 ? args[1] : 0xffff
	  };
      }
    }
  *next_labelled = is_label ? 1 : 0;
  /* tidy up*/
  if(args)
    free(args);
  if(lbls)
    free(lbls);
  if(is_label)
    return dest;
  return dest + 1 + extra_words_needed;
}

/**
 * parse the instructions of json for a single function.
 */
instruction_t *parse_instructions(struct json_array_s* json,
				  struct hashmap *fun_name_to_idx,
				  struct hashmap *tmp_map,
				  uint16_t *num_temps,
				  uint16_t *tmp_types,
				  size_t *num_instrs)
{
  size_t insn_len = 32;
  instruction_t *insns = malloc(sizeof(instruction_t) * insn_len);
  struct hashmap *lbl_map = MAKE_HASH_MAP;
  struct hashmap *prt_lbl_map = MAKE_HASH_MAP;
  struct json_array_element_s *tmp = json->start;
  size_t dest = 0;
  uint16_t next_labelled = 0;
  uint16_t num_lbls = 0;
  const char **idx_to_lbl = malloc(sizeof(char*) * json->length);
  for(size_t i = 0; i < json->length; ++i)
    {
      dest = parse_instruction(json_value_as_object(tmp->value), &insns,
			       dest, &insn_len, &next_labelled,
			       lbl_map, tmp_map, prt_lbl_map, fun_name_to_idx,
			       idx_to_lbl, &num_lbls, num_temps, tmp_types);
      tmp = tmp->next;
    }
  *num_instrs = dest;
  /* clean up filler values for labels, and fill in types for print*/
  for(size_t i = 0; i < dest; ++i)
    {
      instruction_t *insn = insns + i;
      switch(get_opcode(*insn))
	{
	case JMP:
	  insn->norm_insn.dest = translate_label(lbl_map,
						 idx_to_lbl,
						 insn->norm_insn.dest);
	  break;
	case BR:
	  insn->br_inst.ltrue = translate_label(lbl_map,
						idx_to_lbl,
						insn->br_inst.ltrue);
	  insn->br_inst.lfalse = translate_label(lbl_map,
						 idx_to_lbl,
						 insn->br_inst.lfalse);
	  break;
	case PHI:
	  {
	    uint16_t num_lbls = insn->phi_inst.num_choices;
	    for(uint16_t j = 0; j < num_lbls; j += 2)
	      {
		++i;
		instruction_t *phi_ext = insns + i;
		phi_ext->phi_ext.lbl1 = translate_label(lbl_map,
							idx_to_lbl,
							phi_ext->phi_ext.lbl1);
		if(j + 1 < num_lbls)
		  phi_ext->phi_ext.lbl2 = translate_label(lbl_map,
							  idx_to_lbl,
							  phi_ext->phi_ext.lbl2);
	      }
	  } break;
	case PRINT:
	  {
	    uint16_t num_args = insn->print_insn.num_prints;
	    insn->print_insn.type1 = tmp_types[insn->print_insn.arg1];
	    for(uint16_t j = 0; j < num_args - 1; j += 2)
	      {
		++i;
		instruction_t *args = insns + i;
		args->print_args.type1 = tmp_types[args->print_args.arg1];
		if(j + 1 < num_args)
		  args->print_args.type2 = tmp_types[args->print_args.arg2];
	      }
	  } break;
	case CALL:
	  i += (insn->call_inst.num_args + 3) / 4;
	  break;
	case LCONST:
	  ++i;
	  break;
	}
    }
  hashmap_free(lbl_map);
  hashmap_free(tmp_map);
  hashmap_free(prt_lbl_map);
  free(idx_to_lbl);
  free(tmp_types);
  return insns;
}

/**
 * parse a function from json.
 * fun_name_to_idx is a map from function names to their indices in the program.
 */
function_t parse_function(struct json_object_s *json, struct hashmap *fun_name_to_idx)
{
  struct hashmap *tmp_map = MAKE_HASH_MAP;
  uint16_t num_temps = 0;
  uint16_t num_args = 0;
  uint16_t num_instrs = 0;
  uint16_t *tmp_types = 0;
  struct json_object_element_s *field = json->start;
  function_t fun;
  struct json_array_s *instrs_json;
  struct json_array_s *args_json = 0;
  struct json_value_s *ret_tp = 0;
  while(field)
    {
      if(strcmp(field->name->string, "name") == 0)
	{
	  const char *str = json_value_as_string(field->value)->string;
	  /* printf("parsing function %s\n", str); */
	  char *fun_nm = malloc(sizeof(char) * (1 + strlen(str)));
	  fun.name = strcpy(fun_nm, str);
	} else if(strcmp(field->name->string, "instrs") == 0)
	{
	  num_instrs = json_value_as_array(field->value)->length;
	  instrs_json = json_value_as_array(field->value);
	} else if(strcmp(field->name->string, "args") == 0)
	{
	  args_json = json_value_as_array(field->value);
	  num_args = args_json->length;
	} else if(strcmp(field->name->string, "type") == 0)
	{
	  ret_tp = field->value;
	}
      field = field->next;
    }
  tmp_types = malloc(sizeof(uint16_t) * (num_args + num_instrs));
  fun.ret_tp = type_of_json_value(ret_tp);
  fun.arg_types = malloc(sizeof(uint16_t) * num_args);
  fun.num_args = 0;
  if(args_json)
    {
      struct json_array_element_s *arg = args_json->start;
      size_t argidx = 0;
      while(arg)
	{
	  struct json_object_element_s *a = json_value_as_object(arg->value)->start;
	  int16_t alias = num_temps;
	  while(a)
	    {
	      const char *str = a->name->string;
	      if(strcmp(str, "name") == 0)
		{
		  hashmap_set(tmp_map, &(hashdat)
			      {.str = json_value_as_string(a->value)->string,
			       .num = num_temps++});
		} else if(strcmp(str, "type") == 0)
		{
		  uint16_t tp = type_of_json_value(a->value);
		  tmp_types[alias] = tp;
		  fun.arg_types[argidx] = tp;
		}
	      a = a->next;
	    }
	  arg = arg->next;
	  ++argidx;
	}
      fun.num_args = argidx;
    }
  size_t num_words;
  fun.insns = parse_instructions(instrs_json, fun_name_to_idx,
				 tmp_map, &num_temps, tmp_types, &num_words);
  fun.num_insns = num_words;
  fun.num_tmps = num_temps;
  return fun;
}


program_t *parse_program(struct json_object_s *json)
{
  struct json_array_s *json_funcs = json->start->value->payload;
  size_t num_funcs = json_funcs->length;
  program_t *prog = malloc(sizeof(program_t) + sizeof(function_t) * num_funcs);
  struct json_array_element_s *json_fun = json_funcs->start;
  struct hashmap *fun_name_to_idx = MAKE_HASH_MAP;
  for(uint16_t i = 0; i < num_funcs; ++i)
    {
      struct json_object_element_s *field =
	json_value_as_object(json_fun->value)->start;
      while(field)
	{
	  if(strcmp(field->name->string, "name") == 0)
	    {
	      const char *nm = json_value_as_string(field->value)->string;
	      hashmap_set(fun_name_to_idx, &(hashdat){.str = nm, .num = i});
	    }
	  field = field->next;
	}
      json_fun = json_fun->next;
    }
  json_fun = json_funcs->start;
  for(size_t i = 0; i < num_funcs; ++i)
    {
      prog->funcs[i] = parse_function(json_value_as_object(json_fun->value),
				      fun_name_to_idx);
      json_fun = json_fun->next;
    }
  prog->num_funcs = num_funcs;
  hashmap_free(fun_name_to_idx);
  return prog;
}
