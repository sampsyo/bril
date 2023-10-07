#include "linear-scan.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __ARM_ARCH



typedef struct interval
{
  int start, end;
  arm_reg_t reg;
  uint16_t temp, stack_loc;
  bool is_reg;
} interval_t;

typedef struct temps_used
{
  int16_t num;
  int tmps[2];
} temps_used_t;

static inline int max(int a, int b)
{
  return a > b ? a : b;
}

static inline temps_used_t norm_temps_used(arm_arg_tagged_t a1, arm_arg_tagged_t a2)
{
  uint16_t num = (a1.type == TMP ? 1 : 0) +
    (a2.type == TMP ? 1 : 0);
  return (temps_used_t)
    {.num = num,
     .tmps[0] = num == 0 ? -1 : (num == 1 && a1.type != TMP ?
			    a2.value.tmp : a1.value.tmp),
     .tmps[1] = num < 2 ? -1 : a2.value.tmp};
}

static inline temps_used_t one_temps_used(arm_arg_tagged_t a)
{
  uint16_t num = a.type == TMP ? 1 : 0;
  return (temps_used_t)
    {.num = num,
     .tmps[0] = num == 0 ? -1 : a.value.tmp,
     .tmps[1] = -1};
}

temps_used_t get_temps_used(tagged_arm_insn_t insn)
{
  switch(insn.type)
    {
    case ANORM:
      return norm_temps_used(insn.value.norm.a1, insn.value.norm.a2);
    case ACMP:
      return norm_temps_used(insn.value.cmp.a1, insn.value.cmp.a2);
    case AMOV:
      return one_temps_used(insn.value.mov.src);
    case ACBZ:
      return one_temps_used(insn.value.cbz.cond);
    case ASTR:
      return norm_temps_used(insn.value.str.address, insn.value.str.value);
    case ALDR:
      return one_temps_used(insn.value.ldr.address);
    default:
      return (temps_used_t)
	{.num = 0,
	 .tmps[0] = -1,
	 .tmps[1] = -1};
    }
}

temps_used_t get_temps_defd(tagged_arm_insn_t insn)
{
  switch(insn.type)
    {
    case ANORM:
      return one_temps_used(insn.value.norm.dest);
    case ASET:
      return one_temps_used(insn.value.set.dest);
    case AMOV:
      return one_temps_used(insn.value.mov.dest);
    case AMOVC:
      return one_temps_used(insn.value.movc.dest);
    case ALDR:
      return one_temps_used(insn.value.ldr.dest);
    default:
      return (temps_used_t)
	{.num = 0,
	 .tmps[0] = -1,
	 .tmps[1] = -1};
    }
}

int cmp_interval_start(const void *p1, const void *p2)
{
  return ((const interval_t *) p1)->start -
    ((const interval_t *) p2)->start;
}

int cmp_interval_end(const void *p1, const void *p2)
{
  return ((const interval_t *) p1)->end -
    ((const interval_t *) p2)->end;
}


interval_t *get_intervals(asm_func_t f)
{
  interval_t *intervals = malloc(sizeof(interval_t) * f.num_temps);
  printf("%s:\n", f.name);
  for(uint16_t i = 0; i < f.num_temps; ++i)
    {
      intervals[i].temp = i;
      intervals[i].start = -1;
      intervals[i].end = -1;
    }
  for(size_t i = 0; i < f.num_args; ++i)
    {
      intervals[i].start = 0;
    }
  for(int i = 0; i < f.num_insns; ++i)
    {
      if(f.insns[i].type == AABSCALL)
	{
	  if(intervals[f.insns[i].value.abs_call.dest].start == -1)
	    intervals[f.insns[i].value.abs_call.dest].start = i;
	  uint16_t num_args = f.insns[i].value.abs_call.num_args;
	  for(size_t x = 0; x < num_args; x += 32)
	      {
		for(size_t argi = 0; x * 32 + argi < num_args && argi < 32; ++argi)
		  {
		    intervals[f.insns[i + 1 + x/32].value
			      .abs_call_ext.typed_temps[argi] & 0xffff].end = max
		      (intervals[f.insns[i + 1 + x/32].value
				 .abs_call_ext.typed_temps[argi] & 0xffff].end, i);
		    if(intervals[f.insns[i + 1 + x/32].value
				 .abs_call_ext.typed_temps[argi] & 0xffff].start == -1)
		      intervals[f.insns[i + 1 + x/32].value
				.abs_call_ext.typed_temps[argi] & 0xffff].start = i;
		  }
		++i;
              }
	} else
	{
	  temps_used_t t = get_temps_used(f.insns[i]);
	  for(size_t j = 0; j < t.num; ++j)
	    {
	      intervals[t.tmps[j]].end = max(intervals[t.tmps[j]].end, i);
	      if(intervals[t.tmps[j]].start == -1)
		intervals[t.tmps[j]].start = i;
	    }
	  temps_used_t defd = get_temps_defd(f.insns[i]);
	  for(size_t j = 0; j < defd.num; ++j)
	    {
	      if(intervals[defd.tmps[j]].start == -1)
		intervals[defd.tmps[j]].start = i;
	    }
	}
    }
  qsort(intervals, f.num_temps, sizeof(interval_t), cmp_interval_start);
  return intervals;
}


typedef struct reg_pool
{
  size_t top;
  arm_reg_t regs[32];
} reg_pool_t;


bool is_float_reg(arm_reg_t r)
{
  switch (r)
    {
    case D0:
    case D1:
    case D2:
    case D3:
    case D4:
    case D5:
    case D6:
    case D7:
    case D8:
    case D9:
    case D10:
    case D11:
    case D12:
    case D13:
    case D14:
    case D15:
    case D16:
    case D17:
    case D18:
    case D19:
    case D20:
    case D21:
    case D22:
    case D23:
    case D24:
    case D25:
    case D26:
    case D27:
    case D28:
    case D29:
    case D30:
    case D31: return true;
    default: return false;
    }
}

void expire_old_intervals(interval_t i, interval_t *active, size_t *num_active,
			  reg_pool_t *float_pool, reg_pool_t *int_pool)
{
  qsort(active, *num_active, sizeof(interval_t), cmp_interval_end);
  size_t active_shift = 0;
  for(size_t j = 0; j < num_active; ++j)
    {
      if(active[j].end >= i.start)
	return;
      active_shift = j;
      if(is_float_reg(active[j].reg))
	float_pool->regs[++float_pool->top] = active[j].reg;
      else
	int_pool->regs[++int_pool->top] = active[j].reg;
    }
  *num_active -= active_shift;
  memmove(active, active + active_shift * sizeof(interval_t), *num_active);
}

asm_func_t lin_alloc(asm_prog_t p, size_t which_fun)
{
  interval_t *intervals = get_intervals(p.funcs[which_fun]);
  interval_t *active = malloc(sizeof(interval_t) * 128);
  reg_pool_t float_pool = (reg_pool_t)
    {.top = 30,
     .regs = {D31, D30, D29, D28, D27, D26, D25, D24, D23, D22, D21, D20, D19, D18,
       D17, D16, D15, D14, D13, D12, D11, D10, D9, D8, D7, D6, D5, D4, D3, D2,}};
  reg_pool_t int_pool = (reg_pool_t)
    {.top = 29,
     .regs = {X30, X29, X28, X27, X26, X25, X24, X23, X22, X21, X20, X19, X18,
       X17, X16, X15, X14, X13, X12, X11, X10, X9, X8, X7, X6, X5, X4, X3, X2,}};
  
  size_t num_active = 0;
  for(size_t i = 0; i < p.funcs[which_fun].num_temps; ++i)
    {
      printf("t%d: live from %d to %d\n", intervals[i].temp, intervals[i].start, intervals[i].end);
    }
  free(intervals);
  return p.funcs[which_fun];
}

asm_prog_t linear_scan(asm_prog_t p)
{
  asm_func_t *funs = malloc(sizeof(asm_func_t) * p.num_funcs);
  for(size_t i = 0; i < p.num_funcs; ++i)
    funs[i] = lin_alloc(p, i);
  return (asm_prog_t)
    {.funcs = funs,
     .num_funcs = p.num_funcs};
}

#else

asm_prog_t linear_scan(asm_prog_t p)
{
  fprintf(stderr, "arch not supported\n");
  exit(1);
}


#endif
