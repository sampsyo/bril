#include "brili.hpp"

#include <cstdio>
#include <stdexcept>

#include "util/charconv.hpp"
#include "util/unreachable.hpp"

namespace bril {

Brili::Brili(Prog& p, std::ostream& out, std::ostream& err)
    : prog_(p), main_(nullptr), out_(out), err_(err) {
  for (auto& fn : prog_.fns) {
    if (fn.name == "main") {
      main_ = &fn;
      break;
    }
  }
}

struct BrilException {
  bool printStackTrace;
  BrilException(bool printStackTrace_ = true) : printStackTrace(printStackTrace_) {}
};

std::ostream& operator<<(std::ostream& os, const Val& v) {
  // if pointer type, print address (interpreted as pointer)
  if (v.type.ptr_dims) {
    os << "0x" << std::hex << v.i << std::dec;
    return os;
  }

  switch (v.type.kind) {
  case TypeKind::Int:
    os << v.i;
    break;
  case TypeKind::Float:
    os << v.f;
    break;
  case TypeKind::Bool:
    os << v.b;
    break;
  case TypeKind::Char:
    os << v.c;
    break;
  case TypeKind::Void:
    assert(false);
    bril::unreachable();
  }
  return os;
}

void Brili::setLocal(uint32_t dst, Val val) { top().locals[dst] = val; }

const Val& Brili::getLocal(uint32_t src) const {
  if (!top().isInit(src)) {
    err_ << "Use of uninitialized variable." << std::endl;
    throw BrilException();
  }
  return top().locals[src];
}

#define VALUE_BINOP(op, oper, val_case)              \
  case Op::op: {                                     \
    assert(instr.args().size() == 2);                \
    auto left = getLocal(instr.args()[0]).val_case;  \
    auto right = getLocal(instr.args()[1]).val_case; \
    auto val = left oper right;                      \
    setLocal(instr.dst(), Val(val));                 \
    break;                                           \
  }

#define VALUE_BINOP_DIV(op, oper, val_case)          \
  case Op::op: {                                     \
    assert(instr.args().size() == 2);                \
    auto left = getLocal(instr.args()[0]).val_case;  \
    auto right = getLocal(instr.args()[1]).val_case; \
    if (right == 0) {                                \
      err_ << "Division by zero." << std::endl;      \
      throw BrilException();                         \
    }                                                \
    auto val = left oper right;                      \
    setLocal(instr.dst(), Val(val));                 \
    break;                                           \
  }

ActRec makeActRec(std::vector<Val>& args, Func& fn, uint32_t ret_var = 0) {
  ActRec ar{fn, ret_var};
  for (size_t i = 0; i < args.size(); ++i) {
    ar.locals[fn.args[i].name] = args[i];
  }
  return ar;
}

Func* Brili::findFunc(const std::string_view& name) const noexcept {
  for (auto& fn : prog_.fns) {
    if (fn.name == name) return &fn;
  }
  return nullptr;
}

void Brili::execCall(const Instr& call) {
  auto fn_name = top().fn.sp->get(call.func());
  auto fn = findFunc(std::move(fn_name));
  assert(fn);

  auto& args = call.args();
  assert(fn->args.size() == args.size());

  // create activation record for fn
  std::vector<Val> args_vals;
  for (auto arg : args) args_vals.push_back(getLocal(arg));
  stack_.push(makeActRec(args_vals, *fn, call.dst()));
}

void Brili::execRet(const Instr& ret) {
  assert((ret.args().size() == 1 && !top().fn.ret_type.isVoid()) ||
         (ret.args().empty() && top().fn.ret_type.isVoid()));

  // assign to the ret var if present
  if (top().ret_var) {
    auto ret_var = top().ret_var;
    Val ret_val = getLocal(ret.args()[0]);
    stack_.pop();
    assert(!stack_.empty());

    setLocal(ret_var, ret_val);
    ++top().instr;
  }
  // otherwise just pop the frame and return
  else {
    stack_.pop();
    if (!stack_.empty()) ++top().instr;
  }
}

void Brili::exec(const Instr& instr) {
  switch (instr.op()) {
  case Op::Const:
    setLocal(instr.dst(), Val(cast<Const>(instr).lit(), instr.type()));
    break;
    VALUE_BINOP(Mul, *, i);
    VALUE_BINOP(Add, +, i);
    VALUE_BINOP(Sub, -, i);
    VALUE_BINOP_DIV(Div, /, i);
    VALUE_BINOP(Eq, ==, i);
    VALUE_BINOP(Lt, <, i);
    VALUE_BINOP(Le, <=, i);
    VALUE_BINOP(Gt, >, i);
    VALUE_BINOP(Ge, >=, i);
    VALUE_BINOP(And, &&, b);
    VALUE_BINOP(Or, ||, b);
  case Op::Not: {
    auto val = !getLocal(instr.args()[0]).b;
    setLocal(instr.dst(), Val(val));
    break;
  }
  case Op::Id:
    setLocal(instr.dst(), top().locals[instr.args()[0]]);
    break;
  case Op::Call_v:
  case Op::Call_e:
    execCall(instr);
    return;
  case Op::Phi:
    break;
  case Op::Jmp:
    top().instr = 0;
    top().bb = &top().getBB(instr.labels()[0]);
    return;
  case Op::Br:
    top().instr = 0;
    if (getLocal(instr.args()[0]).b) {
      top().bb = &top().getBB(instr.labels()[0]);
    } else {
      top().bb = &top().getBB(instr.labels()[1]);
    }
    return;
  case Op::Ret:
    execRet(instr);
    return;
  case Op::Print:
    for (auto& arg : instr.args()) out_ << getLocal(arg) << " ";
    out_ << std::endl;
    break;
  case Op::Nop:
    break;

  case Op::Store:
  case Op::Free:
  case Op::Alloc:
  case Op::Load:
  case Op::PtrAdd:
  case Op::F_add:
  case Op::F_mul:
  case Op::F_sub:
  case Op::F_div:
  case Op::F_eq:
  case Op::F_lt:
  case Op::F_le:
  case Op::F_gt:
  case Op::F_ge:
  case Op::Speculate:
  case Op::Commit:
  case Op::Guard:
  case Op::C_eq:
  case Op::C_lt:
  case Op::C_le:
  case Op::C_gt:
  case Op::C_ge:
  case Op::Char2int:
  case Op::Int2Char:
    err_ << "Unsupported instruction: " << instr.op() << std::endl;
    throw BrilException(true);
  case Op::Label:
  case Op::KIND_MASK:
    assert(false);
    bril::unreachable();
  }

  ++top().instr;
}

bool Brili::run(std::vector<Val>& args) {
  try {
    if (main_ == nullptr) {
      err_ << "Could not find a main function." << std::endl;
      throw BrilException(false);
    }
    if (main_->args.size() != args.size()) {
      err_ << "Expected " << main_->args.size() << " arguments, got " << args.size()
           << "." << std::endl;
      throw BrilException(false);
    }
    if (!main_->ret_type.isVoid()) {
      err_ << "Expected main to not have a return type." << std::endl;
      throw BrilException(false);
    }

    // create activation record for main
    stack_.push(makeActRec(args, *main_));

    while (true) {
      exec(top().bb->code()[top().instr]);
      if (stack_.empty()) break;

      // at the end of this basic block
      if (top().instr == top().bb->code().size()) {
        // last bb of the function, implicit return
        if (&top().fn.bbs.back() == top().bb) {
          assert(top().fn.ret_type.isVoid());
          stack_.pop();
          if (stack_.empty()) break;
        }
        // otherwise go to the next one
        top().bb = top().bb->nextBB();
        top().instr = 0;
      }
    }

    return true;
  } catch (const BrilException& e) {
    // if (e.printStackTrace) printStackTrace();
    return false;
  }
}

};  // namespace bril