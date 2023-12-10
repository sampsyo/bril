#include "types.hpp"
#include <memory>

namespace bril {
std::vector<const Instr *> Func::allInstrs() const {
  std::vector<const Instr *> res;
  for (auto &bb : bbs) {
    if (bb.label)
      res.push_back(bb.label);
    for (auto &phi : bb.phis)
      res.push_back(&phi);
    for (auto &instr : bb.code)
      res.push_back(&instr);
  }
  return res;
}

void Func::populateBBsV() {
  if (bbsv)
    return;
  bbsv = std::make_unique<BasicBlock *[]>(bbs.size());
  auto bbsvit = &bbsv[0];
  for (auto &bb : bbs)
    *(bbsvit++) = &bb;
}

void Func::deleteBBsV() { bbsv.reset(); }
}; // namespace bril