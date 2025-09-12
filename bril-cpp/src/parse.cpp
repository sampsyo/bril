#include "parse.hpp"

#include <iostream>
#include <memory>

#include "cfg.hpp"
#include "util/charconv.hpp"
#include "util/unreachable.hpp"

namespace bril {

Func* from_fn = nullptr;
VarPool* from_vp = nullptr;

std::string jsonToStr(const json& j) { return j.template get<std::string>(); }

Type type_from_json(const json& j) {
  if (j.is_object()) {
    Type t = type_from_json(j.at("ptr"));
    ++t.ptr_dims;
    return t;
  }

  const auto str = jsonToStr(j);
  if (str == "int")
    return Type::intType();
  else if (str == "bool")
    return Type::boolType();
  else if (str == "float")
    return Type::floatType();
  else if (str == "char")
    return Type::charType();
  assert(false);
}

Op stringToOp(const std::string& s) {
#define OPS_DEF(op, str) \
  if (s == str) return Op::op;
#define NO_CALL_E_CASE
#include "ops.defs"
#undef OPS_DEF
  assert(false);
  bril::unreachable();
}

void from_json(const json& j, Type& t) { t = type_from_json(j); }
ConstLit constLitFromJson(const json& j, Type t) {
  switch (t.kind) {
  case TypeKind::Bool:
    return ConstLit(j.at("value").template get<bool>());
  case TypeKind::Int:
    return ConstLit(j.at("value").template get<int64_t>());
  case TypeKind::Float:
    return ConstLit(j.at("value").template get<double>());
  case TypeKind::Char:
    // FIXME: handle unicode characters
    return ConstLit(static_cast<char32_t>(jsonToStr(j.at("value"))[0]));
  default:
    break;
  }
  bril::unreachable();
  assert(false);
}
Const* const_from_json(const json& j) {
  auto dest = from_vp->varRefOf(jsonToStr(j.at("dest")));
  auto t = type_from_json(j.at("type"));
  auto lit = constLitFromJson(j, t);
  return new Const(t, dest, lit);
}

void addArgs(const json& j, Instr& e) {
  if (!j.contains("args")) return;
  for (auto& a : j.at("args")) {
    e.args().push_back(from_vp->varRefOf(jsonToStr(a)));
  }
}
void addLabels(const json& j, Instr& e) {
  if (!j.contains("labels")) return;
  for (auto& l : j.at("labels")) {
    e.labels().push_back(from_fn->sp->canonicalize(jsonToStr(l)));
  }
}
void addFunc(const json& j, Instr& e) {
  if (!j.contains("funcs")) return;
  e.func() = from_fn->sp->canonicalize(jsonToStr(j.at("funcs")[0]));
}

Instr* instr_from_json(const json& j) {
  if (j.contains("label")) {
    auto name = from_fn->sp->canonicalize(jsonToStr(j.at("label")));
    return new Label(name);
  }
  Op op = stringToOp(jsonToStr(j.at("op")));

  if (op == Op::Const) return const_from_json(j);

  Instr* i;
  if (j.contains("dest")) {
    auto dest = from_vp->varRefOf(j.at("dest").template get<std::string>());
    i = new Value(op, dest, type_from_json(j.at("type")));
  } else {
    if (op == Op::Call_v) op = Op::Call_e;
    i = new Effect(op);
  }

  addArgs(j, *i);
  addFunc(j, *i);
  addLabels(j, *i);
  return i;
}
void from_json(const json& j, Instr*& i) { i = instr_from_json(j); }
void from_json(const json& j, Func& fn) {
  from_fn = &fn;
  from_vp = &fn.vp;

  j.at("name").get_to(fn.name);
  if (j.contains("type")) j.at("type").get_to(fn.ret_type);
  if (j.contains("args")) {
    for (const auto& arg : j.at("args")) {
      auto arg_ref = from_vp->varRefOf(arg.at("name").template get<std::string>());
      fn.args.emplace_back(arg_ref, type_from_json(arg.at("type")));
    }
  }
  std::vector<Instr*> instrs;
  for (const auto& instr : j.at("instrs")) {
    instrs.push_back(instr.template get<Instr*>());
  }
  fn.bbs = toCFG(fn, instrs);
  fn.populateBBsV();

  from_vp = nullptr;
  from_fn = nullptr;
}
void from_json(const json& j, Prog& p) { j.at("functions").get_to(p.fns); }

const Func* to_fn = nullptr;
const VarPool* to_vp = nullptr;

void to_json(json& j, Type t) {
  switch (t.kind) {
  case TypeKind::Int:
    j = "int";
    break;
  case TypeKind::Bool:
    j = "bool";
    break;
  case TypeKind::Float:
    j = "float";
    break;
  case TypeKind::Char:
    j = "char";
    break;
  case TypeKind::Void:
  default:
    assert(false);
    bril::unreachable();
  }
  for (int i = 0; i < t.ptr_dims; ++i) {
    j = json{{"ptr", j}};
  }
}
void to_json(json& j, Arg const& a) {
  j = json{{"name", to_vp->strOf(a.name)}, {"type", a.type}};
}

std::vector<std::string> labelsToStrs(const LabelVec& refs) {
  std::vector<std::string> views;
  for (auto r : refs) views.push_back(bbIdToNameStr(*to_fn, r));
  return views;
}

void to_json(json& j, const Value& i) {
  j = json{
      {"dest", to_vp->strOf(i.dst())}, {"op", toString(i.op())}, {"type", i.type()}};
  if (!i.args().empty()) {
    std::vector<std::string_view> args;
    for (auto a : i.args()) args.push_back(to_vp->strOf(a));
    j["args"] = std::move(args);
  }
  if (i.op() == Op::Call_v) j["funcs"] = json{to_fn->sp->get(i.func())};
  if (!i.labels().empty()) {
    j["labels"] = labelsToStrs(i.labels());
  }
}
void to_json(json& j, const Label& i) { j = json{{"label", to_fn->sp->get(i.name())}}; }
void to_json(json& j, const Effect& i) {
  j = json{{"op", toString(i.op())}};
  if (!i.args().empty()) {
    std::vector<std::string_view> args;
    for (auto a : i.args()) args.push_back(to_vp->strOf(a));
    j["args"] = std::move(args);
  }
  if (i.op() == Op::Call_e) j["funcs"] = json{to_fn->sp->get(i.func())};
  if (!i.labels().empty()) j["labels"] = labelsToStrs(i.labels());
}
void const_lit_to_json(json& j, const ConstLit& lit, Type type) {
  switch (type.kind) {
  case TypeKind::Bool:
    j["value"] = lit.bool_val;
    break;
  case TypeKind::Int:
    j["value"] = lit.int_val;
    break;
  case TypeKind::Float:
    j["value"] = lit.fp_val;
    break;
  case TypeKind::Char:
    j["value"] = toString(lit.char_val);
    break;
  default:
    assert(false);
    bril::unreachable();
  }
}
void to_json(json& j, const Const& i) {
  j = json{{"dest", to_vp->strOf(i.dst())}, {"op", "const"}, {"type", i.type()}};
  const_lit_to_json(j, i.lit(), i.type());
}
void to_json(json& j, const Instr& i) {
  if (i.op() == Op::Label) {
    to_json(j, cast<Label>(i));
    return;
  }
  switch (opKindMasked(i.op())) {
  case Op::Const:
    to_json(j, cast<Const>(i));
    break;
  case Op::VALUE_MASK:
    to_json(j, cast<Value>(i));
    break;
  case Op::EFFECT_MASK:
    to_json(j, cast<Effect>(i));
    break;
  default:
    assert(false);
    bril::unreachable();
  }
}
void to_json(json& j, const Instr& i, const Func& fn) {
  to_fn = &fn;
  to_vp = &fn.vp;

  to_json(j, i);

  to_vp = nullptr;
  to_fn = nullptr;
}
json to_json(const Instr& i, const Func& fn) {
  json j;
  to_json(j, i, fn);
  return j;
}
void to_json(json& j, const Instr* i) { to_json(j, *i); }
void to_json(json& j, const Func& fn) {
  to_fn = &fn;
  to_vp = &fn.vp;

  json instrs = json::array();
  for (const auto& bb : fn.bbs) {
    std::string name = bbNameToStr(*to_fn, bb);
    instrs.push_back({{"label", name}});
    for (const auto& phi : bb.phis()) instrs.push_back(phi);
    for (const auto& instr : bb.code()) instrs.push_back(instr);
  }

  j = json{{"name", fn.name}, {"args", fn.args}, {"instrs", std::move(instrs)}};
  if (!fn.ret_type.isVoid()) j["type"] = fn.ret_type;

  to_vp = nullptr;
  to_fn = nullptr;
}
void to_json(json& j, const Prog& p) { j = json{{"functions", p.fns}}; }

}  // namespace bril