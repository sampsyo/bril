#include <iostream>
#include <nlohmann/json.hpp>

#include "brili.hpp"
#include "parse.hpp"

using json = nlohmann::json;

int main() {
  json j = json::parse(std::cin);
  auto prog = j.template get<bril::Prog>();
  bril::Brili brili(prog);
  std::vector<bril::Val> args;
  brili.run(args);
}