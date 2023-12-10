#pragma once

#include "types.hpp"

namespace bril {

BBList toCFG(std::vector<Instr *> &instrs);

} // namespace bril