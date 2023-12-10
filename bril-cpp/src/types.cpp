#include "types.hpp"

#include <memory>
#include <ostream>
#include <sstream>

#include "unreachable.hpp"

namespace bril {
std::vector<const Instr*> Func::allInstrs() const noexcept {
  std::vector<const Instr*> res;
  for (auto& bb : bbs) {
    for (auto& phi : bb.phis()) res.push_back(&phi);
    for (auto& instr : bb.code()) res.push_back(&instr);
  }
  return res;
}

void Func::populateBBsV() noexcept {
  if (bbsv) return;
  bbsv = std::make_unique<std::vector<BasicBlock*>>();
  bbsv->reserve(bbs.size());
  for (auto& bb : bbs) bbsv->push_back(&bb);
}

void Func::deleteBBsV() noexcept { bbsv.reset(); }

std::string toString(Op op) {
  std::ostringstream ss;
  ss << op;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, Op op) {
  switch (op) {
#define OPS_DEF(x, s) \
  case Op::x:         \
    return os << s;
#include "ops.defs"
#undef OPS_DEF
  default:
    assert(false);
    bril::unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const Type& t) {
  switch (t.kind) {
  case TypeKind::Int:
    os << "int";
    break;
  case TypeKind::Bool:
    os << "bool";
    break;
  case TypeKind::Char:
    os << "char";
    break;
  case TypeKind::Float:
    os << "float";
    break;
  default:
    assert(false);
    bril::unreachable();
  }
  for (uint32_t i = 0; i < t.ptr_dims; ++i) os << "*";
  return os;
}

std::string bbNameToStr(const Func& fn, const BasicBlock& bb) noexcept {
  if (bb.name()) return std::string(fn.sp->get(bb.name()));
  return "_bb." + std::to_string(bb.id());
}
std::string bbIdToNameStr(const Func& fn, uint32_t bb) noexcept {
  return bbNameToStr(fn, *(*fn.bbsv)[bb]);
}
};  // namespace bril