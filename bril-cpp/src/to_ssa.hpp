#pragma once

#include <boost/dynamic_bitset.hpp>
#include <stack>
#include <unordered_set>

#include "types.hpp"

namespace bril {
class ToSSA {
  Func& fn_;
  BBList& bbs_;
  const unsigned int nbbs_;
  const unsigned int nvars_;
  bool remove_dead_phis_;

  struct VarInfo;

  // variable to blocks that define the variables
  std::vector<VarInfo> var_infos_;

  // variable stack for renaming
  std::stack<VarRef>* name_stacks_;
  std::vector<VarRef>* name_logs_;

  VarRef getNewName(VarRef var);
  VarRef genNewName(VarRef var);

 public:
  ToSSA(Func& fn, bool remove_dead_phis);
  ~ToSSA();

  void rename(BasicBlock& bb);
  // rename the variables to fully convert to SSA. Call after addPhiNodes();
  void rename();

  // find variable definition to basic block mappings (and uses)
  void computeVarInfos();

  // compute the iterated dominance frontier for var and put the result in phis
  void idf(unsigned int var, boost::dynamic_bitset<>& temp,
           boost::dynamic_bitset<>& phis);
  // remove dead phis; that is, phis who are never used
  void removeDeadPhis(unsigned int var, boost::dynamic_bitset<>& phis);
  // add phi nodes to the CFG. Call after dominanceTree() and findDefs() have
  // been computed
  void addPhiNodes();

  void toSSA();
};
}  // namespace bril
