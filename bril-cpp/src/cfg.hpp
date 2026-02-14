#pragma once

#include "types.hpp"

namespace bril {

BBList toCFG(Func& fn, std::vector<Instr*>& instrs);

}  // namespace bril