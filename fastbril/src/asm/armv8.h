#ifndef ARMV8_H
#define ARMV8_H
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/**
 * "normal" operations in armv8 (prefixed w/ A for "arm")
 */
typedef enum norm_arm_op
  {
    AADD, AMUL, ASUB, AAND, AORR, ASDIV, ALSL, AFADD, AFMUL, AFSUB, AFDIV,
  } norm_arm_op_t;

typedef enum arm_reg
  {
    SP, X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30,
    XZR, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15,
    D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30,
    D31,
  } arm_reg_t;

typedef enum arm_cmp_flags
  {
    CMPEQ, CMPMI, CMPLS, CMPGT, CMPGE, CMPLT, CMPLE,
  } arm_cmp_flags_t;

typedef enum arm_arg_tp {REG, CNST, TMP} arm_arg_tp_t;

typedef union arm_arg
{
  arm_reg_t reg;
  int32_t cnst;
  uint16_t tmp;
} arm_arg_t;

typedef struct arm_arg_tagged
{
  arm_arg_tp_t type;
  arm_arg_t value;
} arm_arg_tagged_t;

typedef struct norm_arm_insn
{
  norm_arm_op_t op;
  arm_arg_tagged_t dest, a1, a2;
  uint8_t lsl;
} norm_arm_insn_t;

typedef struct cmp_arm_insn
{
  bool is_float;
  arm_arg_tagged_t a1, a2;
} cmp_arm_insn_t;

typedef struct set_arm_insn
{
  arm_arg_tagged_t dest;
  arm_cmp_flags_t flag;
} set_arm_insn_t;

typedef struct mov_arm_insn
{
  arm_arg_tagged_t dest, src;
  bool is_float;
} mov_arm_insn_t;

typedef struct movk_arm_insn
{
  arm_arg_tagged_t dest;
  uint16_t val;
  uint16_t lsl;
} movk_arm_insn_t;

typedef struct movc_arm_insn
{
  arm_arg_tagged_t dest;
  int64_t val;
} movc_arm_insn_t;

typedef struct cbz_arm_insn
{
  arm_arg_tagged_t cond;
  char dest[100];
} cbz_arm_insn_t;

typedef struct call_arm_insn
{
  char name[126];
} call_arm_insn_t;

typedef struct abs_call_arm_insn
{
  uint16_t num_args;
  uint16_t ret_tp;
  uint16_t dest;
  char name[100];
} abs_call_arm_insn_t;

typedef struct abs_call_arm_ext
{
  /* n & 0xffff is the temp. n >> 16 is the type */
  uint32_t typed_temps[32];
} abs_call_arm_ext_t;

typedef struct str_arm_insn
{
  arm_arg_tagged_t value, address;
  uint16_t offset;
} str_arm_insn_t;

typedef struct ldr_arm_insn
{
  arm_arg_tagged_t dest, address;
  uint16_t offset;
} ldr_arm_insn_t;

typedef enum arm_insn_type
  {
    ANORM, ACMP, ASET, AMOV, AMOVK, AOTHER, ACBZ, ACALL, AABSCALL,
    AABSEXT, ASTR, ALDR, AMOVC,
  } arm_insn_type_t;

typedef union arm_insn
{
  norm_arm_insn_t norm;
  cmp_arm_insn_t cmp;
  set_arm_insn_t set;
  mov_arm_insn_t mov;
  movk_arm_insn_t movk;
  movc_arm_insn_t movc;
  cbz_arm_insn_t cbz;
  call_arm_insn_t call;
  abs_call_arm_insn_t abs_call;
  abs_call_arm_ext_t abs_call_ext;
  ldr_arm_insn_t ldr;
  str_arm_insn_t str;
  char other[128];
} arm_insn_t;

typedef struct tagged_arm_insn
{
  arm_insn_type_t type;
  arm_insn_t value;
} tagged_arm_insn_t;

typedef tagged_arm_insn_t asm_insn_t;


#endif
