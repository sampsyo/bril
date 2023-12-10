#include "dom.hpp"

#include <queue>
#include <unordered_set>

#include "types.hpp"

namespace bril {

#define TO_UINT static_cast<unsigned int>

DomAnalysis::DomAnalysis(Func& fn) : bbsa_(*fn.bbsv), n_(fn.bbs.size()) {
  for (auto& bb : fn.bbs) {
    bb.domInfo() = new DomInfo(n_);
  }
}

struct DomValue {
  BasicBlock* bb;

  // iterate the dataflow analysis
  // out = {n} union (intersect of n'.out for n' in preds)
  // returns true updated
  bool iter(DomValue* vals, boost::dynamic_bitset<>& temp);
};

bool DomValue::iter(DomValue* vals, boost::dynamic_bitset<>& temp) {
  auto& out = bb->domInfo()->dom_by;
  // use temp so we don't have to allocate any new bitsets
  auto& preds = bb->entries();
  temp = vals[preds[0]->id()].bb->domInfo()->dom_by;
  for (auto it = ++preds.begin(); it != preds.end(); it++)
    temp &= vals[(*it)->id()].bb->domInfo()->dom_by;

  temp.set(TO_UINT(bb->id()));
  if (out == temp) return false;
  out.swap(temp);

  return true;
}

DomValue* createDVArray(std::vector<BasicBlock*>& bbs) {
  auto res = new DomValue[bbs.size()];
  for (size_t i = 0; i < bbs.size(); i++) res[i].bb = &*bbs[i];
  return res;
}

void destroyDFArray(DomValue* x) { delete[] x; }

// computes dominated_by set for each of the basic blocks.
void DomAnalysis::computeDomBy() {
  // initialize dataflow values
  auto vals = createDVArray(bbsa_);

  // initialize data flow worklist
  std::queue<unsigned int> wl;
  boost::dynamic_bitset<> in_wl(n_);

  auto add_to_wl = [&in_wl, &wl](BasicBlock& bb) {
    for (auto exit : bb.exits()) {
      if (!exit) break;
      auto e_id = TO_UINT(exit->id());
      if (!in_wl.test_set(e_id)) {
        wl.push(e_id);
      }
    }
  };

  // set value for first bb
  bbsa_[0]->domInfo()->dom_by.reset();
  bbsa_[0]->domInfo()->dom_by.set(0);
  in_wl.set(0);  // don't allow first bb to be processed
  add_to_wl(*bbsa_[0]);

  // df analysis
  boost::dynamic_bitset<> temp(n_);
  while (!wl.empty()) {
    auto& next = vals[wl.front()];
    auto& next_bb = next.bb;
    in_wl.reset(wl.front());
    wl.pop();
    // if true, more work to do
    if (next.iter(vals, temp)) {
      // if this node changed, add all successors to worklist
      add_to_wl(*next_bb);
    }
  }
}

void DomAnalysis::computeDomTree() {
  boost::dynamic_bitset<> temp(n_);
  for (size_t i = 0; i < n_; i++) {
    auto bb = bbsa_[i];
    for (unsigned int j = 0; j < n_; j++) {
      if (bb->domInfo()->dom_by.test(j) && TO_UINT(bb->id()) != j) {
        auto dom = bbsa_[j];
        temp.reset();
        // dom is bb's idom iff
        // dom's dominators are equal to bb's dominators - bb
        temp |= dom->domInfo()->dom_by;
        temp.set(TO_UINT(bb->id()));
        if (temp == bb->domInfo()->dom_by) {
          bb->domInfo()->idom = dom;
          dom->domInfo()->succs.push_back(bb);
          break;
        }
      }
    }
  }
}

void domTreeGV(Func& fn, std::ostream& os) {
  os << "digraph ";
  os << fn.name;
  os << " {\nnode[shape=rectangle]\n";
  os << "label=";
  os << '"' << fn.name << '"';
  os << ";\n";
  for (auto& bb : fn.bbs) {
    auto name = bbNameToStr(fn, bb);
    os << '"' << name << '"';
    os << " [label=\"";
    os << name;
    os << "\"];\n";
  }

  for (auto& bb : fn.bbs) {
    auto bb_name = bbNameToStr(fn, bb);
    for (auto dom : bb.domInfo()->succs) {
      os << '"' << bb_name << '"';
      os << "->";
      os << '"' << bbNameToStr(fn, *dom) << '"';
      os << ";\n";
    }
  }
  os << "}\n";
}

void DomAnalysis::computeDomFront() { domFrontHelper(*bbsa_[0]); }

void DomAnalysis::domFrontHelper(BasicBlock& bb) {
  auto& df_n = bb.domInfo()->dfront;

  // {n' | n ≻ n'}
  for (auto exit : bb.exits()) {
    if (!exit) break;
    df_n.set(TO_UINT(exit->id()));
  }

  // U_{n idom c} DF[c]
  for (auto bb2 : bb.domInfo()->succs) {
    domFrontHelper(*bb2);
    df_n |= bb2->domInfo()->dfront;
  }

  // {n' | n dom n'}
  for (unsigned int i = 0; i < n_; i++) {
    if (!df_n.test(i) || i == TO_UINT(bb.id())) continue;
    auto front = bbsa_[i];
    if (front->domInfo()->dom_by.test(TO_UINT(bb.id()))) df_n.reset(i);
  }
}

}  // namespace bril
