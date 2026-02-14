#include "brili.hpp"

#include <cstdio>
#include <stdexcept>

#include "util/charconv.hpp"
#include "util/unreachable.hpp"

namespace bril {

Brili::Brili(Prog& p, std::ostream& out, std::ostream& err)
    : prog_(p), main_(nullptr), heap_(err), out_(out), err_(err) {
  for (auto& fn : prog_.fns) {
    if (fn.name == "main") {
      main_ = &fn;
      break;
    }
  }
}

struct BrilException {};

std::ostream& operator<<(std::ostream& os, const Val& v) {
  // if pointer type, print address (interpreted as pointer)
  if (v.type.ptr_dims) {
    os << "0x" << std::hex << v.p.value() << std::dec;
    return os;
  }

  switch (v.type.kind) {
  case TypeKind::Int:
    os << v.i;
    break;
  case TypeKind::Float:
    if (v.f == std::numeric_limits<double>::infinity())
      os << "Infinity";
    else if (v.f == -std::numeric_limits<double>::infinity())
      os << "-Infinity";
    else if (std::isnan(v.f))
      os << "NaN";
    else
      os << v.f;
    break;
  case TypeKind::Bool:
    os << (v.b ? "true" : "false");
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

Ptr Heap::alloc(size_t sz) noexcept {
  uint32_t key;
  if (!free_list_.empty()) {
    key = free_list_.back();
    free_list_.pop_back();
    data_[key].freed = false;
  } else {
    key = static_cast<uint32_t>(data_.size());
    data_.emplace_back();
  }
  data_[key].data.resize(sz);
  return {0, key};
}
void Heap::free(Ptr ptr) {
  if (ptr.key >= data_.size() || ptr.offset) {
    cerr_ << "error: attempted to free invalid ptr " << ptr.value() << std::endl;
    throw BrilException();
  }
  if (data_[ptr.key].freed) {
    cerr_ << "error: attempted to free already freed ptr " << ptr.value() << std::endl;
    throw BrilException();
  }
  data_[ptr.key].freed = true;
  free_list_.push_back(ptr.key);
}
void Heap::checkPtr(Ptr ptr) const {
  if (ptr.key >= data_.size()) {
    cerr_ << "error: attempted to access to invalid ptr " << ptr.value() << std::endl;
    throw BrilException();
  }
  if (data_[ptr.key].freed) {
    cerr_ << "error: attempted to access to freed ptr " << ptr.value() << std::endl;
    throw BrilException();
  }
}
void Heap::store(Ptr ptr, Val val) {
  checkPtr(ptr);
  data_[ptr.key].data[ptr.offset] = val;
}
Val Heap::load(Ptr ptr) const {
  checkPtr(ptr);
  return data_[ptr.key].data[ptr.offset];
}

void Brili::setLocal(uint32_t dst, Val val) { top().locals[dst] = val; }

const Val& Brili::getLocal(uint32_t src) const {
  if (!top().isInit(src)) {
    err_ << "error: use of uninitialized variable." << std::endl;
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

#define VALUE_BINOP_DIV(op, oper, val_case)            \
  case Op::op: {                                       \
    assert(instr.args().size() == 2);                  \
    auto left = getLocal(instr.args()[0]).val_case;    \
    auto right = getLocal(instr.args()[1]).val_case;   \
    if (right == 0) {                                  \
      err_ << "error: division by zero." << std::endl; \
      throw BrilException();                           \
    }                                                  \
    auto val = left oper right;                        \
    setLocal(instr.dst(), Val(val));                   \
    break;                                             \
  }

ActRec makeActRec(std::vector<Val>& args, Func& fn, uint32_t ret_var = 0) {
  assert(fn.args.size() == args.size());
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
  ++top().instr;
  auto fn_name = top().fn.sp->get(call.func());
  auto fn = findFunc(std::move(fn_name));
  assert(fn);

  auto& args = call.args();
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
  }
  // otherwise just pop the frame and return
  else {
    stack_.pop();
  }
}

void Brili::setBB(uint32_t bb) noexcept {
  top().last_bb = static_cast<uint32_t>(top().bb->id());
  top().bb = &top().getBB(bb);
  top().instr = 0;
  execPhis();
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
  case Op::Call_v:
  case Op::Call_e:
    execCall(instr);
    return;
  case Op::Jmp:
    setBB(instr.labels()[0]);
    return;
  case Op::Br:
    if (getLocal(instr.args()[0]).b) {
      setBB(instr.labels()[0]);
    } else {
      setBB(instr.labels()[1]);
    }
    return;
  case Op::Ret:
    execRet(instr);
    return;
  case Op::Print:
    for (auto it = instr.args().begin(); it != std::prev(instr.args().end()); it++) {
      out_ << getLocal(*it) << " ";
    }
    out_ << getLocal(instr.args().back());
    out_ << std::endl;
    break;
  case Op::Nop:
    break;

  case Op::Id:
  case Op::Char2int:
  case Op::Int2Char:
    setLocal(instr.dst(), top().locals[instr.args()[0]]);
    break;

    VALUE_BINOP(F_add, +, f);
    VALUE_BINOP(F_mul, *, f);
    VALUE_BINOP(F_sub, -, f);
    VALUE_BINOP(F_div, /, f);
    VALUE_BINOP(F_eq, ==, f);
    VALUE_BINOP(F_lt, <, f);
    VALUE_BINOP(F_le, <=, f);
    VALUE_BINOP(F_gt, >, f);
    VALUE_BINOP(F_ge, >=, f);

    VALUE_BINOP(C_eq, ==, c);
    VALUE_BINOP(C_lt, <, c);
    VALUE_BINOP(C_le, <=, c);
    VALUE_BINOP(C_gt, >, c);
    VALUE_BINOP(C_ge, >=, c);

  case Op::Store: {
    auto ptr = getLocal(instr.args()[0]).p;
    auto val = getLocal(instr.args()[1]);
    heap_.store(ptr, val);
    break;
  }
  case Op::Free: {
    auto ptr = getLocal(instr.args()[0]).p;
    heap_.free(ptr);
    break;
  }
  case Op::Alloc: {
    auto sz = getLocal(instr.args()[0]).i;
    auto ptr = heap_.alloc(static_cast<size_t>(sz));
    setLocal(instr.dst(), Val(ptr, instr.type()));
    break;
  }
  case Op::Load: {
    auto ptr = getLocal(instr.args()[0]).p;
    auto val = heap_.load(ptr);
    setLocal(instr.dst(), val);
    break;
  }
  case Op::PtrAdd: {
    auto ptr = getLocal(instr.args()[0]).p;
    auto offset = getLocal(instr.args()[1]).i;
    auto new_ptr = ptr + offset;
    setLocal(instr.dst(), Val(new_ptr, instr.type()));
    break;
  }

  case Op::Speculate:
  case Op::Commit:
  case Op::Guard:
    err_ << "Unsupported instruction: " << instr.op() << std::endl;
    throw BrilException();

  case Op::Label:
  case Op::KIND_MASK:
  case Op::Phi:  // phi should only be handled when entering a new bb
    assert(false);
    bril::unreachable();
  }

  ++top().instr;
}

struct SetPrecision {
  std::ostream& os;
  std::streamsize old_prec;
  std::ios_base::fmtflags old_flags;
  SetPrecision(std::ostream& os_, std::streamsize prec)
      : os(os_),
        old_prec(os.precision(prec)),
        old_flags(os_.setf((os_.flags() | os_.fixed) & ~os_.scientific)) {}
  ~SetPrecision() {
    os.precision(old_prec);
    os.setf(old_flags);
  }
};

size_t findPhiIdx(const Instr& phi, uint32_t bb_id) {
  auto l_it = std::find(phi.labels().begin(), phi.labels().end(), bb_id);
  assert(l_it != phi.labels().end());
  return static_cast<size_t>(std::distance(phi.labels().begin(), l_it));
}

void Brili::execPhis() {
  auto& bb = *top().bb;
  if (bb.phis().empty()) return;

  auto idx = findPhiIdx(bb.phis().front(), static_cast<uint32_t>(top().last_bb));

  for (auto& phi : bb.phis()) {
    auto& val = getLocal(phi.args()[idx]);
    setLocal(phi.dst(), val);

    ++res_.total_dyn_inst;
  }
}

Result Brili::run(std::vector<Val>& args) {
  SetPrecision p(out_, 17);
  // FIXME: clear stack and heap

  res_ = Result();
  try {
    assert(main_);
    // create activation record for main
    stack_.push(makeActRec(args, *main_));

    while (!stack_.empty()) {
      // at the top in case we jump to an empty block
      // at the end of this basic block
      while (top().instr == top().bb->code().size()) {
        // last bb of the function, implicit return
        if (top().bb == &top().fn.bbs.back()) {
          assert(top().fn.ret_type.isVoid());
          stack_.pop();
          if (stack_.empty()) break;
        } else {
          // otherwise go to the next one
          setBB(static_cast<uint32_t>(top().bb->id() + 1));
        }
      }
      if (stack_.empty()) break;

      ++res_.total_dyn_inst;
      exec(top().bb->code()[top().instr]);
    }

    return res_;
  } catch (const BrilException& e) {
    // printStackTrace();
    return res_;
  }
}
};  // namespace bril