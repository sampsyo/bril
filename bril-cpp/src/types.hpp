#pragma once

#include <boost/intrusive/list.hpp>
#include <list>
#include <ostream>
#include <string>
#include <vector>

#include "casting.hpp"
#include "small_vector.hpp"
#include "string_pool.hpp"

namespace bril {
// TYPE

enum class TypeKind : char { Void, Int, Bool, Float, Char };

struct Type {
  TypeKind kind : 5;
  uint32_t ptr_dims : 27;

  Type(TypeKind kind_) : kind(kind_), ptr_dims(0) {}
  Type(TypeKind kind_, uint32_t ptr_dims_) : kind(kind_), ptr_dims(ptr_dims_) {}

  static Type intType(uint32_t ptr_dims = 0) noexcept;
  static Type boolType(uint32_t ptr_dims = 0) noexcept;
  static Type floatType(uint32_t ptr_dims = 0) noexcept;
  static Type charType(uint32_t ptr_dims = 0) noexcept;
  static Type voidType() noexcept;

  bool isVoid() const noexcept { return kind == TypeKind::Void; }
};
std::ostream& operator<<(std::ostream& os, const Type& t);

// INSTRUCTION

enum class Op : uint16_t {
  EFFECT_MASK = 0x0800,
  VALUE_MASK = 0x8000,
  KIND_MASK = EFFECT_MASK | VALUE_MASK,

  Const = 0,

  // CORE VALUES
  Add = 0x8000,  // 32768
  Mul,
  Sub,
  Div,
  Eq,
  Lt,
  Gt,
  Le,
  Ge,
  Not,
  And,
  Or,
  Id,
  Call_v,

  Phi,  // 327872

  Jmp = 0x0C00,  // 3072
  Br,            // 3073

  Call_e = 0x0800,  // 2048
  Ret,

  Print,
  Nop,  // 2051

  Alloc = 0xC000,  // 49152
  Free,
  Store,
  Load,
  PtrAdd,  // 49156

  F_add = 0xA000,  // 40960
  F_mul,
  F_sub,
  F_div,
  F_eq,
  F_lt,
  F_le,
  F_gt,
  F_ge,  // 40968

  Speculate = 0x0A00,  // 2560
  Commit,
  Guard,  // 2562

  C_eq = 0x9000,  // 36864
  C_lt,
  C_le,
  C_gt,
  C_ge,
  Char2int,
  Int2Char,  // 36870

  Label = 1,  // 1
};

inline auto opToInt(Op op) { return static_cast<std::underlying_type_t<Op>>(op); }
inline auto intToOp(std::underlying_type_t<Op> op) { return static_cast<Op>(op); }
inline auto opKindMasked(Op op) {
  return intToOp(opToInt(op) & opToInt(Op::KIND_MASK));
}
std::string toString(Op op);
std::ostream& operator<<(std::ostream& os, Op op);

union ConstLit {
  int64_t int_val;
  bool bool_val;
  double fp_val;
  uint32_t char_val;

  ConstLit(int64_t val) : int_val(val) {}
  ConstLit(bool val) : bool_val(val) {}
  ConstLit(double val) : fp_val(val) {}
  ConstLit(uint32_t val) : char_val(val) {}
};

enum class InstrKind : char { Label, Const, Value, Effect };

using ArgVec = bril::SmallVector<VarRef, 2>;
using LabelRef = StringRef;
using LabelVec = bril::SmallVector<LabelRef, 2>;

struct Instr : public boost::intrusive::list_base_hook<> {
  const InstrKind kind;

  VarRef dst_;
  Type type_;
  Op op_;
  ConstLit lit_;
  ArgVec args_;
  LabelVec labels_;

  ConstLit& lit() noexcept { return lit_; }
  const ConstLit& lit() const noexcept { return lit_; }
  Op op() const noexcept { return op_; }

  const ArgVec& args() const noexcept { return args_; }
  ArgVec& args() noexcept { return args_; }

  const LabelVec& labels() const noexcept { return labels_; }
  LabelVec& labels() noexcept { return labels_; }

  const VarRef& dst() const { return dst_; }
  VarRef& dst() { return dst_; }

  const Type& type() const noexcept;
  Type& type() noexcept;

  bool isJump() const noexcept;
  bool isPhi() const noexcept;

 protected:
  Instr(const InstrKind kind_, Op op)
      : kind(kind_), type_(Type::voidType()), op_(op), lit_(0LL) {}
  Instr(const InstrKind kind_, Op op, Type type)
      : kind(kind_), type_(type), op_(op), lit_(0LL) {}
  Instr(const InstrKind kind_, Type type, VarRef dst, ConstLit lit)
      : kind(kind_), dst_(dst), type_(type), op_(Op::Const), lit_(lit) {}
};

struct Label : Instr {
  LabelRef name;

  Label(LabelRef name_) : Instr(InstrKind::Label, Op::Label), name(name_) {}

  static bool classof(const Instr* t) { return t->kind == InstrKind::Label; }
};

struct Const : Instr {
  Const(Type type, VarRef dst, ConstLit lit)
      : Instr(InstrKind::Const, type, dst, lit) {}

  static bool classof(const Instr* t) { return t->kind == InstrKind::Const; }
};

struct Value : Instr {
  std::vector<std::string> funcs;

  Value(Op op, VarRef dst_, Type type) : Instr(InstrKind::Value, op, type) {
    this->dst_ = dst_;
  }

  static bool classof(const Instr* t) { return t->kind == InstrKind::Value; }
};

struct Effect : Instr {
  std::vector<std::string> funcs;

  Effect(Op op) : Instr(InstrKind::Effect, op) {}

  static bool classof(const Instr* t) { return t->kind == InstrKind::Effect; }
};

using InstrList = boost::intrusive::list<Instr>;

// BASIC BLOCKS

struct DomInfo;

struct BasicBlock : public boost::intrusive::list_base_hook<> {
  // serial number of basic block in the function
  int id;
  StringRef name;
  // predecessors of this basic block in the cfg
  std::vector<BasicBlock*> entries;
  // successors of this basic block in the cfg
  BasicBlock* exits[2] = {nullptr, nullptr};
  // dominator info
  DomInfo* dom_info;

  // potentially contains a label that begins this basic block
  Label* label = nullptr;
  // contains all phi nodes
  InstrList phis;
  // contains instructions
  InstrList code;

  BasicBlock(StringRef name_) : id(-1), name(name_) {}
  BasicBlock(int id_, StringRef name_) : id(id_), name(name_) {}
};

using BBList = boost::intrusive::list<BasicBlock>;

// FUNCTION AND PROGRAM

struct Arg {
  VarRef name;
  Type type;

  Arg(VarRef name_, Type type_) : name(name_), type(type_) {}
};

struct Func {
  std::string name;
  Type ret_type{TypeKind::Void};
  std::vector<Arg> args;
  BBList bbs;
  std::unique_ptr<BasicBlock*[]> bbsv;

  StringPool* sp;
  VarPool vp;

  std::vector<const Instr*> allInstrs() const;
  void populateBBsV();
  void deleteBBsV();

  Func() : sp(new StringPool()), vp(*sp) {}
};

struct Prog {
  std::vector<Func> fns;
};

}  // namespace bril

namespace bril {

inline const Type& Instr::type() const noexcept { return type_; }
inline Type& Instr::type() noexcept {
  return const_cast<Type&>(const_cast<const Instr*>(this)->type());
}

inline bool Instr::isJump() const noexcept { return op_ == Op::Jmp || op_ == Op::Br; }

inline bool Instr::isPhi() const noexcept { return op_ == Op::Phi; }

inline Type Type::intType(uint32_t ptr_dims) noexcept {
  return Type(TypeKind::Int, ptr_dims);
}
inline Type Type::boolType(uint32_t ptr_dims) noexcept {
  return Type(TypeKind::Bool, ptr_dims);
}
inline Type Type::floatType(uint32_t ptr_dims) noexcept {
  return Type(TypeKind::Float, ptr_dims);
}
inline Type Type::charType(uint32_t ptr_dims) noexcept {
  return Type(TypeKind::Char, ptr_dims);
}
inline Type Type::voidType() noexcept { return Type(TypeKind::Void); }
}  // namespace bril
