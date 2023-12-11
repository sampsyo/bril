#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>

#include "brili.hpp"
#include "parse.hpp"

using json = nlohmann::json;

bril::Val parseAs(std::string_view s, bril::Type t) {
  switch (t.kind) {
  case bril::TypeKind::Int:
    return bril::Val(std::stoll(std::string(s)));
  case bril::TypeKind::Float:
    return bril::Val(std::stod(std::string(s)));
  case bril::TypeKind::Bool:
    return bril::Val(s == "true");
  case bril::TypeKind::Char:
    return bril::Val(static_cast<char32_t>(s[0]));
  default:
    bril::unreachable();
  }
}

int main(int argc, char** argv) {
  std::vector<std::string_view> sv_args;
  bool print_total_dyn_inst = false;
  for (int i = 1; i < argc; i++) {
    if (argv[i] == std::string_view("-p")) {
      print_total_dyn_inst = true;
      continue;
    }
    sv_args.emplace_back(argv[i]);
  }

  json j = json::parse(std::cin);
  auto prog = j.template get<bril::Prog>();

  bril::Func* main = nullptr;
  for (bril::Func& fn : prog.fns) {
    if (fn.name == "main") {
      main = &fn;
      break;
    }
  }

  if (main == nullptr) {
    std::cerr << "error: could not find a main function." << std::endl;
    return 1;
  }
  if (main->args.size() != sv_args.size()) {
    std::cerr << "error: expected " << main->args.size() << " arguments, got "
              << sv_args.size() << std::endl;
    return 1;
  }
  if (!main->ret_type.isVoid()) {
    std::cerr << "error: expected main to not have a return type" << std::endl;
    return 1;
  }

  std::vector<bril::Val> args;
  for (size_t i = 0; i < sv_args.size(); i++) {
    try {
      args.emplace_back(parseAs(sv_args[i], main->args[i].type));
    } catch (std::invalid_argument& e) {
      std::cerr << "error: could not parse argument " << sv_args[i] << " as "
                << main->args[i].type << std::endl;
      return 1;
    }
  }

  bril::Brili brili(prog);
  brili.run(args);

  if (print_total_dyn_inst) {
    std::cerr << "total_dyn_inst: " << 15 << std::endl;
  }
}