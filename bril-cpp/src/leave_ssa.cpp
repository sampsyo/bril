#include "leave_ssa.hpp"

#include <iostream>
#include <queue>
#include <string>

namespace bril {
#define TO_UINT static_cast<unsigned int>

void addEdgeBlock(Func& fn, BasicBlock& bb, BasicBlock*& exit) {
  auto edge_block = new BasicBlock(static_cast<int>(fn.bbs.size()), 0);
  fn.bbs.push_back(*edge_block);

  // add moves at the edge to set the phi destination with the var defined in bb
  for (auto& phi : exit->phis()) {
    auto it = std::find(phi.labels().begin(), phi.labels().end(), bb.id());
    // incoming name that the phi uses
    auto inc_name = phi.args()[TO_UINT(std::distance(phi.labels().begin(), it))];
    Value mov(Op::Id, phi.dst(), phi.type());
    mov.args().push_back(inc_name);
    edge_block->code().push_back(mov);
  }

  // update entries by removing old bb and adding new one
  exit->entries().erase(std::find(exit->entries().begin(), exit->entries().end(), &bb));
  exit->entries().push_back(edge_block);

  // unconditional jump from edge_block to exit
  edge_block->exits()[0] = exit;
  Effect jmp(Op::Jmp);
  jmp.labels().push_back(TO_UINT(exit->id()));
  edge_block->code().push_back(std::move(jmp));
  // set the exit to edge_block
  exit = edge_block;
  edge_block->entries().push_back(&bb);
}

void leaveSSA(Func& fn) {
  size_t n = fn.bbs.size();
  for (unsigned int i = 0; i < n; i++) {
    auto& bb = *(*fn.bbsv)[i];
    for (int j = 0; j < 2; j++) {
      auto& exit = bb.exits()[j];
      if (!exit) break;
      // don't need to insert edge block or change jump labels
      if (exit->phis().empty()) continue;

      addEdgeBlock(fn, bb, exit);
    }

    bb.fixTermLabels();
    bb.addTermIfNotPresent();
  }

  // remove phi nodes from all the bbs
  for (unsigned int i = 0; i < n; i++) {
    (*fn.bbsv)[i]->phis().clear();
  }

  fn.deleteBBsV();
}
}  // namespace bril