#pragma once

#include <cassert>
#include <nlohmann/json.hpp>

#include "types.hpp"
#include "unreachable.hpp"

namespace bril {

using json = nlohmann::json;

ConstLit constLitFromJson(const json& j, Type t);
Type type_from_json(const json& j);
void from_json(const json& j, Type& t);
Const* const_from_json(const json& j);
Instr* instr_from_json(const json& j);
void from_json(const json& j, Instr*& i);
void from_json(const json& j, Func& fn);
void from_json(const json& j, Prog& p);

void to_json(json& j, const Instr& i, const Func& fn);
json to_json(const Instr& i, const Func& fn);
void to_json(json& j, const Func& fn);
void to_json(json& j, const Prog& p);

}  // namespace bril
