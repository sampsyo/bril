#include "types.hpp"

#include <memory>
#include <ostream>
#include <sstream>

#include "unreachable.hpp"

namespace bril {
std::vector<const Instr*> Func::allInstrs() const {
  std::vector<const Instr*> res;
  for (auto& bb : bbs) {
    for (auto& phi : bb.phis) res.push_back(&phi);
    for (auto& instr : bb.code) res.push_back(&instr);
  }
  return res;
}

void Func::populateBBsV() {
  if (bbsv) return;
  bbsv = std::make_unique<BasicBlock*[]>(bbs.size());
  auto bbsvit = &bbsv[0];
  for (auto& bb : bbs) *(bbsvit++) = &bb;
}

void Func::deleteBBsV() { bbsv.reset(); }

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
};  // namespace bril