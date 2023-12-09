#ifndef TYPES_HPP
#define TYPES_HPP

#include <boost/intrusive/list.hpp>
#include <cstdint>
#include <list>
#include <string>
#include <vector>

#include "casting.hpp"
#include "string_pool.hpp"

namespace bril {

enum class TypeKind : char { Int, Bool, Float, Char, Ptr };

struct Type {
  TypeKind kind : 5;
  uint32_t ptr_dims : 27;

  Type(TypeKind kind_) : kind(kind_), ptr_dims(0) {}
  Type(TypeKind kind_, uint32_t ptr_dims_) : kind(kind_), ptr_dims(ptr_dims_) {}
};

union ConstLit {
  int64_t int_val;
  bool bool_val;
  double fp_val;
  uint32_t char_val;

  ConstLit(long long val) : int_val(val) {}
  ConstLit(bool val) : bool_val(val) {}
  ConstLit(double val) : fp_val(val) {}
  ConstLit(uint32_t val) : char_val(val) {}
};

enum class Op : uint16_t {
  EFFECT_MASK = 0x0800,
  VALUE_MASK = 0x8000,

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
  Int2Char  // 36870
};

inline auto opToInt(Op op) { return static_cast<std::underlying_type_t<Op>>(op); }

struct BasicBlock;

using ArgIt = uint32_t*;
using ConstArgIt = const uint32_t*;

using LabelIt = uint16_t*;
using ConstLabelIt = const uint16_t*;

struct Instr {
 protected:
  union {
    struct {
      Type type_;
      uint32_t dst_;
      uint16_t label_start_;
    };
    uint16_t labels_[2];
  };
  Op op_;

  uint32_t id_;

  union {
    uint32_t args_[2];
    ConstLit lit_;
    struct {
      uint32_t arg_start_;
      uint32_t n_args_;
    };
  };

  friend struct BasicBlockBuilder;
  friend struct CodeEditor;
  friend struct InstrEditor;

 protected:
  Instr(Op op, uint32_t id) : op_(op), id_(id) {}

 public:
  Op op() const noexcept { return op_; }

  template <typename T>
  T as() {
    return cast<T>(this);
  }

  ArgIt argsBegin(BasicBlock&) noexcept;
  ArgIt argsEnd(BasicBlock&) noexcept;
  ConstArgIt argsCBegin(const BasicBlock&) noexcept;
  ConstArgIt argsCEnd(const BasicBlock&) noexcept;

  LabelIt labelBegin(BasicBlock&) noexcept;
  LabelIt labelEnd(BasicBlock&) noexcept;
  ConstLabelIt labelCBegin(const BasicBlock&) const noexcept;
  ConstLabelIt labelCEnd(const BasicBlock&) const noexcept;

  uint32_t& dst() noexcept { return dst_; }
  const uint32_t& dst() const noexcept { return dst_; }
};

struct Const : public Instr {
 public:
  ConstLit& literal() noexcept { return lit_; }
  ConstLit literal() const noexcept { return lit_; }

  static bool classof(const Instr* i) { return i->op() == Op::Const; }

  Const(Op op, uint32_t id, Type type, uint32_t dst, ConstLit lit);
};

struct Value : public Instr {
 public:
  static bool classof(const Instr* i) {
    return opToInt(i->op()) & opToInt(Op::VALUE_MASK);
  }

  Value(Op op, uint32_t id, Type type, uint32_t dst);
  Value(Op op, uint32_t id, Type type, uint32_t dst, uint32_t arg0, uint32_t arg1 = 0);
};

struct Effect : public Instr {
 public:
  static bool classof(const Instr* i) {
    return opToInt(i->op()) & opToInt(Op::EFFECT_MASK);
  }

  Effect(Op op, uint32_t id);
  Effect(Op op, uint32_t id, uint16_t label0, uint16_t label1 = 0);
  Effect(Op op, uint32_t id, uint32_t arg0, uint32_t arg1 = 0);
  Effect(Op op, uint32_t id, uint32_t arg0, uint16_t label0, uint16_t label1 = 0);
};

struct BasicBlock : public boost::intrusive::list_base_hook<> {
 private:
  uint32_t id_;
  uint32_t name_;

  std::vector<Instr> phis_;
  std::vector<Instr> instrs_;
  std::vector<uint32_t> args_;
  std::vector<uint16_t> labels_;

  uint32_t exits_[2];

 public:
  const uint32_t& id() const noexcept { return id_; }
  uint32_t& id() noexcept { return id_; }

  const uint32_t& name() const noexcept { return name_; }
  uint32_t& name() noexcept { return name_; }

  const uint32_t* exits() const noexcept { return exits_; }
  uint32_t* exits() noexcept { return exits_; }

 private:
  BasicBlock(uint32_t id, uint32_t name, uint32_t exit0 = 0, uint32_t exit1 = 0)
      : id_(id), name_(name) {
    exits_[0] = exit0;
    exits_[1] = exit1;
  }

  friend struct BasicBlockBuilder;
  friend struct CodeEditor;
};

// BASIC BLOCKS

using BBList = boost::intrusive::list<BasicBlock>;

// FUNCTION AND PROGRAM

struct Arg {
  VarRef name;
  Type* type;

  Arg(VarRef name_, Type* type_) : name(name_), type(type_) {}
};

struct Func {
  std::string name;
  Type* ret_type;
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

inline Const::Const(Op op, uint32_t id, Type type, uint32_t dst, ConstLit lit)
    : Instr(op, id) {
  this->dst_ = dst;
  this->lit_ = lit;
  this->type_ = type;
}

inline Value::Value(Op op, uint32_t id, Type type, uint32_t dst) : Instr(op, id) {
  this->dst_ = dst;
  this->type_ = type;
}

inline Value::Value(Op op, uint32_t id, Type type, uint32_t dst, uint32_t arg0,
                    uint32_t arg1)
    : Instr(op, id) {
  this->dst_ = dst;
  this->type_ = type;
  this->args_[0] = arg0;
  this->args_[1] = arg1;
}

inline Effect::Effect(Op op, uint32_t id) : Instr(op, id) {}

inline Effect::Effect(Op op, uint32_t id, uint16_t label0, uint16_t label1)
    : Instr(op, id) {
  this->labels_[0] = label0;
  this->labels_[1] = label1;
}

inline Effect::Effect(Op op, uint32_t id, uint32_t arg0, uint32_t arg1)
    : Instr(op, id) {
  this->args_[0] = arg0;
  this->args_[1] = arg1;
}

inline Effect::Effect(Op op, uint32_t id, uint32_t arg0, uint16_t label0,
                      uint16_t label1)
    : Instr(op, id) {
  this->args_[0] = arg0;
  this->labels_[0] = label0;
  this->labels_[1] = label1;
}

}  // namespace bril

#endif  // TYPES_HPP
