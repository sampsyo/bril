#include <iostream>
#include <nlohmann/json.hpp>

#include "dom.hpp"
#include "parse.hpp"
#include "types.hpp"

using json = nlohmann::json;

int main() {
  json j = json::parse(std::cin);
  auto prog = j.template get<bril::Prog>();

  for (auto& fn : prog.fns) {
    bril::DomAnalysis doma(fn);
    doma.computeDomBy();
    auto& bbsa = *fn.bbsv;

    // prints the dom by set for each bb
    std::cout << fn.name << '\n';
    for (auto& bb : fn.bbs) {
      std::cout << bril::bbNameToStr(fn, bb) << ": ";
      for (size_t i = 0; i < fn.bbs.size(); i++) {
        if (bb.domInfo()->dom_by.test(i)) {
          std::cout << bril::bbNameToStr(fn, *bbsa[i]) << ", ";
        }
      }
      std::cout << '\n';
    }
    std::cout << '\n';

    doma.computeDomTree();
    bril::domTreeGV(fn, std::cout);
    std::cout << '\n';

    doma.computeDomFront();
    for (auto& bb : fn.bbs) {
      std::cout << bril::bbNameToStr(fn, bb) << ": ";
      for (size_t i = 0; i < fn.bbs.size(); i++) {
        if (bb.domInfo()->dfront.test(i)) {
          std::cout << bril::bbNameToStr(fn, *bbsa[i]) << ", ";
        }
      }
      std::cout << '\n';
    }

    std::cout << std::endl;
  }
}