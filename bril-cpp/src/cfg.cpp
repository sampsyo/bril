#include "cfg.hpp"

#include <iostream>
#include <unordered_map>

#include "casting.hpp"
#include "types.hpp"

namespace bril {

bool is_term(const Instr* i) { return i->isJump(); }

BBList* cur_bbs = nullptr;
std::unordered_map<LabelRef, BasicBlock*>* bb_map = nullptr;
Func* cur_fn = nullptr;

StringRef canon(std::string s) { return cur_fn->sp->canonicalize(std::move(s)); }

void formBBs(std::vector<Instr*>& instrs) {
  auto cur = new BasicBlock(0, 0);
  cur_bbs->push_back(*cur);
  int bb_cnt = 1;
  for (auto instr : instrs) {
    if (auto label = dyn_cast<Label>(instr)) {
      cur = new BasicBlock(bb_cnt, label->name());
      cur_bbs->push_back(*cur);
      bb_map->emplace(std::make_pair(label->name(), &cur_bbs->back()));
      bb_cnt++;
      continue;
    }

    if (instr->isPhi())
      cur->phis().push_back(*instr);
    else
      cur->code().push_back(*instr);

    if (is_term(instr)) {
      cur = new BasicBlock(bb_cnt, 0);
      cur_bbs->push_back(*cur);
      bb_cnt++;
    }
  }
}

void deleteEmptyBBs() {
  // don't delete first or last bb
  auto it = --(--cur_bbs->end());
  while (it != cur_bbs->begin()) {
    auto& bb = *it;
    if (bb.code().empty()) {
      if (bb.name()) {
        // remap this bb to the next one
        (*bb_map)[bb.name()] = &*std::next(it);
      }
      it = cur_bbs->erase(it);
    }
    --it;
  }
  //   for (auto it = ++cur_bbs->begin(); std::next(it) != cur_bbs->end();) {
  //     auto& bb = *it;
  //     if (bb.code().empty()) {
  //       if (bb.name()) {
  //         // remap this bb to the next one
  //         (*bb_map)[bb.name()] = &*std::next(it);
  //       }
  //       it = cur_bbs->erase(it);
  //       continue;
  //     }
  //     ++it;
}

void renumberBBs() {
  int i = -1;
  for (auto& bb : *cur_bbs) bb.id() = ++i;
}

void connectBBs() {
  for (auto it = cur_bbs->begin(); it != cur_bbs->end(); ++it) {
    auto& bb = *it;
    if (!bb.code().empty()) {
      auto& last = bb.code().back();
      if (last.op() == Op::Jmp) {
        bb.exits()[0] = (*bb_map)[last.labels()[0]];
        bb.exits()[0]->entries().push_back(&bb);
        continue;
      } else if (last.op() == Op::Br) {
        bb.exits()[0] = (*bb_map)[last.labels()[0]];
        bb.exits()[0]->entries().push_back(&bb);
        bb.exits()[1] = (*bb_map)[last.labels()[1]];
        bb.exits()[1]->entries().push_back(&bb);
        continue;
      }
    }
    auto c = it;
    if (++c == cur_bbs->end()) break;

    bb.exits()[0] = &*c;
    bb.exits()[0]->entries().push_back(&bb);
  }

  // insert empty starting block
  if (cur_bbs->empty()) {
    cur_bbs->push_back(*new BasicBlock(0));
    return;
  }
  //   auto& first = cur_bbs->front();
  //   // need to start with a block with no predecessors
  //   if (!first.entries.empty()) {
  //     assert(first.name);
  //     auto& start = *new BasicBlock(0);
  //     cur_bbs->push_front(start);
  //     start.exits()[0] = &first;
  //     first.entries.push_back(&start);
  //   }
}

// replace all string refs to bb names with bb ids
void replaceStrWithIds() {
  for (auto& bb : *cur_bbs) {
    for (auto& phi : bb.phis()) {
      for (auto& label : phi.labels()) {
        label = static_cast<uint32_t>((*bb_map)[label]->id());
      }
    }

    for (auto& instr : bb.code()) {
      if (instr.hasLabels()) {
        for (auto& label : instr.labels()) {
          label = static_cast<uint32_t>((*bb_map)[label]->id());
        }
      }
    }
  }
}

BBList toCFG(Func& fn, std::vector<Instr*>& instrs) {
  BBList res;
  std::unordered_map<StringRef, BasicBlock*> map;
  bb_map = &map;
  cur_bbs = &res;
  cur_fn = &fn;

  formBBs(instrs);
  deleteEmptyBBs();
  connectBBs();
  renumberBBs();
  replaceStrWithIds();

  bb_map = nullptr;
  cur_bbs = nullptr;
  cur_fn = nullptr;
  return res;
}

}  // namespace bril
