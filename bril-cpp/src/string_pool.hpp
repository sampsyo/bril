#pragma once

#include <cassert>
#include <climits>
#include <string>
#include <unordered_map>
#include <vector>

namespace bril {

using StringRef = unsigned int;
auto constexpr const kStringRefMax = UINT_MAX;

class StringPool {
  // id is stored as register number - kARegStart
  std::unordered_map<std::string, StringRef> str_to_ref_;
  // (start index, size)
  std::vector<std::pair<unsigned int, unsigned int>> ref_to_str_;
  std::vector<char> pool_;

public:
  template <typename Str> StringRef canonicalize(Str &&s);

  std::string_view get(StringRef ref) const;
};

using VarRef = int;
auto constexpr const kTempRefMax = UINT_MAX;

// manages temps: holding their names, creating fresh & SSA temps
class VarPool {
  std::unordered_map<StringRef, VarRef> name_to_id_;
  std::vector<StringRef> id_to_name_;
  StringPool *sp_;
  // maps from a ref to its representative (without the suffix)
  std::vector<VarRef> ref_to_rep_;
  // maps from a ref to the number of its latest
  std::vector<size_t> ref_to_suffix_;

public:
  VarPool(StringPool &sp) : sp_(&sp) {}

  // returns the next numbered temp
  VarRef nextVarOf(VarRef tmp);

  std::string_view strOf(VarRef tmp_id) const;
  StringRef strRefOf(VarRef tmp_id) const;

  VarRef freshVar();
  VarRef varRefOf(StringRef name);
  template <typename Str> VarRef varRefOf(Str &&name);
  VarRef origRefOf(VarRef name);

  size_t nvars() const { return id_to_name_.size(); }
};

} // namespace bril

// implementations
namespace bril {

template <typename Str> StringRef StringPool::canonicalize(Str &&s_) {
  auto s = std::string(std::forward<Str &&>(s_));
  auto it = str_to_ref_.find(s);
  if (it != str_to_ref_.end())
    return it->second;

  auto ref = ref_to_str_.size();
  ref_to_str_.emplace_back(pool_.size(), s.size());
  pool_.insert(pool_.end(), s.begin(), s.end());
  str_to_ref_.insert(it, std::make_pair(s, ref));
  assert(ref_to_str_.size() < kStringRefMax);
  return static_cast<StringRef>(ref);
}

inline std::string_view StringPool::get(StringRef ref) const {
  auto &si = ref_to_str_[ref];
  return std::string_view(&pool_[si.first], si.second);
}

inline VarRef VarPool::nextVarOf(VarRef tmp) {
  auto rep = ref_to_rep_[static_cast<size_t>(tmp)];
  auto suff = ++ref_to_suffix_[static_cast<size_t>(rep)];
  auto id = id_to_name_.size();
  std::string name(strOf(rep));
  name += '.';
  name += std::to_string(suff);
  auto name_ref = sp_->canonicalize(name);
  name_to_id_[name_ref] = static_cast<VarRef>(id);
  id_to_name_.push_back(name_ref);
  ref_to_rep_.push_back(rep);
  ref_to_suffix_.emplace_back();
  return static_cast<VarRef>(id);
}

inline std::string_view VarPool::strOf(VarRef tmp_id) const {
  return sp_->get(strRefOf(tmp_id));
}

inline StringRef VarPool::strRefOf(VarRef tmp_id) const {
  return id_to_name_[static_cast<size_t>(tmp_id)];
}

inline VarRef VarPool::varRefOf(StringRef name) {
  auto it = name_to_id_.find(name);
  if (it != name_to_id_.end())
    return it->second;

  auto id = id_to_name_.size();
  id_to_name_.emplace_back(name);
  name_to_id_.insert(it, std::make_pair(name, id));
  ref_to_rep_.push_back(static_cast<VarRef>(id));
  ref_to_suffix_.emplace_back();
  return static_cast<VarRef>(id);
}

template <typename Str> VarRef VarPool::varRefOf(Str &&name) {
  return varRefOf(sp_->canonicalize(std::forward<Str>(name)));
}

inline VarRef VarPool::freshVar() {
  auto reg_name = sp_->canonicalize("__t" + std::to_string(id_to_name_.size()));
  auto id = id_to_name_.size();
  id_to_name_.emplace_back(reg_name);
  name_to_id_[reg_name] = static_cast<VarRef>(id);
  ref_to_rep_.push_back(static_cast<VarRef>(id));
  ref_to_suffix_.emplace_back();
  return static_cast<VarRef>(id);
}

inline VarRef VarPool::origRefOf(VarRef name) {
  return ref_to_rep_[static_cast<unsigned>(name)];
}

} // namespace bril