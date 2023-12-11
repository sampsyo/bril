#include <iostream>
#include <nlohmann/json.hpp>

#include "leave_ssa.hpp"
#include "parse.hpp"
#include "to_ssa.hpp"

using json = nlohmann::json;

int main(int argc, char* argv[]) {
  bool no_leave = false, no_to = false;
  if (argc == 2) {
    std::string arg = argv[1];
    if (arg == "--no-leave") {
      no_leave = true;
    } else if (arg == "--no-to") {
      no_to = true;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  } else {
    if (argc != 1) {
      std::cerr << "Usage: " << argv[0] << " [--no-leave|--no-to]" << std::endl;
      return 1;
    }
  }

  json j = json::parse(std::cin);
  auto prog = j.template get<bril::Prog>();

  if (!no_to) {
    for (auto& fn : prog.fns) {
      bril::ToSSA(fn, true).toSSA();
    }
  }
  if (!no_leave) {
    for (auto& fn : prog.fns) {
      bril::leaveSSA(fn);
      fn.populateBBsV();
    }
  }

  json out = prog;
  std::cout << out << std::endl;
}