#ifndef EDIT_HPP
#define EDIT_HPP

#include "types.hpp"

namespace bril {

struct BasicBlockBuilder;

struct InstrEditor {
 private:
  BasicBlockBuilder& bbb_;
  Instr& instr_;

 public:
  InstrEditor(BasicBlockBuilder& bbb, Instr& instr) : bbb_(bbb), instr_(instr) {}

  Instr& instr() noexcept { return instr_; }

  void addArg(uint32_t arg) noexcept;
  void addLabel(uint16_t label) noexcept;
};

struct BasicBlockBuilder {
 private:
  uint32_t id_;
  uint32_t name_;

  std::vector<Instr> phis_;
  std::vector<Instr> instrs_;
  std::vector<uint32_t> args_;
  std::vector<uint16_t> labels_;

  uint32_t exits_[2];

  friend struct InstrEditor;

 public:
  BasicBlockBuilder() {}

  // Destructor
  ~BasicBlockBuilder() {}

  // Methods
  void buildBasicBlock() {}

  // Other member variables and methods
  InstrEditor addInstr(Instr instr, bool argv = false, bool labelv = false) noexcept;

  uint32_t* exits() noexcept { return exits_; }
};
}  // namespace bril

namespace bril {

inline InstrEditor BasicBlockBuilder::addInstr(Instr instr, bool argv,
                                               bool labelv) noexcept {
  instrs_.push_back(instr);

  // set arg and label start based off argv and labelv flags
  if (argv) instr.arg_start_ = args_.size();
  if (labelv) instr.label_start_ = labels_.size();

  return InstrEditor(*this, instrs_.back());
}

inline void InstrEditor::addArg(uint32_t arg) noexcept {
  instr_.n_args_++;
  bbb_.args_.push_back(arg);
}

inline void InstrEditor::addLabel(uint16_t label) noexcept {
  bbb_.labels_.push_back(label);
}

}  // namespace bril

#endif  // EDIT_HPP
