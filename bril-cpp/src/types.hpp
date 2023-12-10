#pragma once

#include <boost/container/small_vector.hpp>
#include <boost/intrusive/list.hpp>
#include <list>
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

// INSTRUCTION

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

struct Instr : public boost::intrusive::list_base_hook<> {
  const InstrKind kind;

  Type type_;
  ArgVec args_;

  const ArgVec& args() const noexcept { return args_; }
  ArgVec& args() noexcept { return args_; }

  const VarRef* def() const;
  VarRef* def();

  const Type& type() const;
  Type& type();

  bool isJump() const;
  bool isPhi() const;

 protected:
  Instr(const InstrKind kind_) : kind(kind_), type_(Type::voidType()) {}
  Instr(const InstrKind kind_, Type type) : kind(kind_), type_(type) {}
};

struct Label : Instr {
  std::string name;

  Label(std::string&& name_) : Instr(InstrKind::Label), name(std::move(name_)) {}

  static bool classof(const Instr* t) { return t->kind == InstrKind::Label; }
};

struct Const : Instr {
  VarRef dest;
  ConstLit lit;

  Const(VarRef dest_, Type type_, ConstLit lit_)
      : Instr(InstrKind::Const), dest(dest_), lit(lit_) {
    this->type_ = type_;
  }

  static bool classof(const Instr* t) { return t->kind == InstrKind::Const; }
};

struct Value : Instr {
  VarRef dest;
  std::string op;
  std::vector<std::string> labels;
  std::vector<std::string> funcs;

  Value(VarRef dest_, Type type, std::string&& op_)
      : Instr(InstrKind::Value, type), dest(dest_), op(std::move(op_)) {}

  static bool classof(const Instr* t) { return t->kind == InstrKind::Value; }
};

struct Effect : Instr {
  std::string op;
  std::vector<std::string> labels;
  std::vector<std::string> funcs;

  Effect(std::string&& op_) : Instr(InstrKind::Effect), op(std::move(op_)) {}

  static bool classof(const Instr* t) { return t->kind == InstrKind::Effect; }
};

using InstrList = boost::intrusive::list<Instr>;

// BASIC BLOCKS

struct DomInfo;

struct BasicBlock : public boost::intrusive::list_base_hook<> {
  // serial number of basic block in the function
  int id;
  std::string name;
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

  BasicBlock(std::string&& name_) : id(-1), name(std::move(name_)) {}
  BasicBlock(int id_, std::string&& name_) : id(id_), name(std::move(name_)) {}
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

inline const VarRef* Instr::def() const {
  switch (kind) {
  case bril::InstrKind::Const:
    return &cast<Const>(this)->dest;
  case bril::InstrKind::Value:
    return &cast<Value>(this)->dest;
  case bril::InstrKind::Label:
  case bril::InstrKind::Effect:
    return nullptr;
  }
}
inline VarRef* Instr::def() {
  return const_cast<VarRef*>((const_cast<const Instr*>(this)->def()));
}

inline const Type& Instr::type() const { return type_; }
inline Type& Instr::type() {
  return const_cast<Type&>(const_cast<const Instr*>(this)->type());
}

inline bool Instr::isJump() const {
  if (auto eff = dyn_cast<Effect>(this)) {
    return eff->op == "jmp" || eff->op == "br";
  }
  return false;
}

inline bool Instr::isPhi() const {
  if (auto eff = dyn_cast<Value>(this)) return eff->op == "phi";
  return false;
}

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
