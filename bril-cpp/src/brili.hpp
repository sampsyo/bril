#ifndef BRILI_HPP
#define BRILI_HPP

#include <_types/_uint32_t.h>

#include <iostream>
#include <stack>

#include "types.hpp"

namespace bril {

struct Ptr {
  uint32_t offset;
  uint32_t key;

  uint64_t value() const noexcept {
    return (static_cast<uint64_t>(key) << 32) | offset;
  }

  Ptr operator+(int64_t x) noexcept {
    auto v = value() + static_cast<uint64_t>(x);
    return {static_cast<uint32_t>(v & 0xffffffff), static_cast<uint32_t>(v >> 32)};
  }
};

struct Val {
  union {
    int64_t i;
    double f;
    bool b;
    char32_t c;
    Ptr p;
  };
  Type type;

  Val() : i(0) {}
  Val(const ConstLit& lit, Type type_) : i(lit.int_val), type(type_) {}
  Val(int64_t i_) : i(i_), type(TypeKind::Int) {}
  Val(double f_) : f(f_), type(TypeKind::Float) {}
  Val(bool b_) : b(b_), type(TypeKind::Bool) {}
  Val(char32_t c_) : c(c_), type(TypeKind::Char) {}
  Val(Ptr p_, Type type_) : p(p_), type(type_) {}
};
std::ostream& operator<<(std::ostream& os, const Val& v);

// represents an activation record/call frame
struct ActRec {
  const Func& fn;
  const BasicBlock* bb;

  std::vector<Val> locals;
  uint32_t last_bb;
  uint32_t instr;
  uint32_t ret_var;  // return variable that should be set when this frame returns

  ActRec(const Func& fn_, uint32_t ret_ = 0)
      : fn(fn_),
        bb((*fn.bbsv)[0]),
        locals(fn_.vp.nvars(), 0LL),
        last_bb(0),
        instr(0),
        ret_var(ret_) {}

  bool isInit(uint32_t v) const noexcept { return !locals[v].type.isVoid(); }
  BasicBlock& getBB(uint32_t i) const noexcept { return *fn.bbsv->at(i); }
};

struct Result {
  bool success = true;
  size_t total_dyn_inst = 0;
};

struct Heap {
 private:
  struct Entry {
    bool freed = false;
    std::vector<Val> data;
  };
  std::vector<Entry> data_;
  std::vector<uint32_t> free_list_;

  std::ostream& cerr_;

  void checkPtr(Ptr ptr) const;

 public:
  Ptr alloc(size_t sz) noexcept;
  void free(Ptr ptr);
  void store(Ptr ptr, Val val);
  Val load(Ptr ptr) const;

  Heap(std::ostream& cerr) : cerr_(cerr) {}
};

struct Brili {
 private:
  Prog& prog_;
  Func* main_;

  Heap heap_;
  std::stack<ActRec, std::vector<ActRec>> stack_;
  // TODO: heap

  std::ostream& out_;
  std::ostream& err_;

  Result res_;

  Func* findFunc(const std::string_view& name) const noexcept;

  void printStackTrace() const;
  void execRet(const Instr& ret);
  void execCall(const Instr& call);
  void exec(const Instr& instr);
  void execPhis();
  void setLocal(uint32_t dst, Val val);
  void setBB(uint32_t bb) noexcept;
  const Val& getLocal(uint32_t src) const;
  const ActRec& top() const noexcept { return stack_.top(); }
  ActRec& top() noexcept { return stack_.top(); }

 public:
  Brili(Prog& p, std::ostream& out = std::cout, std::ostream& err = std::cerr);

  Result run(std::vector<Val>& args);
};

}  // namespace bril

#endif  // BRILI_HPP
