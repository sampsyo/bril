#ifndef BRILI_HPP
#define BRILI_HPP

#include <_types/_uint32_t.h>

#include <iostream>
#include <stack>

#include "types.hpp"

namespace bril {

struct Val {
  union {
    int64_t i;
    double f;
    bool b;
    char32_t c;
  };
  Type type;

  Val() : i(0) {}
  Val(const ConstLit& lit, Type type_) : i(lit.int_val), type(type_) {}
  Val(int64_t i_) : i(i_), type(TypeKind::Int) {}
  Val(double f_) : f(f_), type(TypeKind::Float) {}
  Val(bool b_) : b(b_), type(TypeKind::Bool) {}
  Val(char32_t c_) : c(c_), type(TypeKind::Char) {}
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

struct Brili {
 private:
  Prog& prog_;
  Func* main_;
  std::stack<ActRec> stack_;
  ActRec* top_;
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
