#include "bril/BrilOps.h"
#include "bril/BrilTypes.h"
#include "bril/MLIRGen.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir::bril;
using namespace bril;

using llvm::dyn_cast;
using llvm::isa;
using llvm::SmallVector;
using llvm::StringRef;

using nlohmann::json;

namespace {
class MLIR2BrilImpl {
public:
  MLIR2BrilImpl() { DEBUG = getenv("DEBUG") != nullptr; }

  json brilGenModule(mlir::ModuleOp module) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenModule\n";

    json brilJson = {{"functions", nlohmann::json::array()}};
    json functions = nlohmann::json::array();

    for (auto fn : module.getOps<mlir::bril::FuncOp>()) {
      auto funcJson = brilGenFunc(fn);
      functions.push_back(std::move(funcJson));
    }

    brilJson["functions"] = functions;

    return brilJson;
  }

private:
  bool DEBUG;
  llvm::DenseMap<mlir::Value, std::string> idMap;
  llvm::DenseMap<mlir::Block *, std::string> blockLabels;

  json getTypeJson(mlir::Type type) {
    if (type.isInteger(64))
      return "int";
    if (type.isInteger(1))
      return "bool";
    if (auto ptrType = dyn_cast<mlir::bril::PtrType>(type)) {
      if (ptrType.getPointeeType().isInteger(64))
        return {{"ptr", "int"}};
      else if (ptrType.getPointeeType().isInteger(1))
        return {{"ptr", "bool"}};
      else {
        llvm::errs() << "Unsupported pointee type in getTypeJson: "
                     << ptrType.getPointeeType() << "\n";
        abort();
      }
    }
    llvm::errs() << "Unsupported type in getTypeString: " << type << "\n";
    abort();
  }

  std::string getId(mlir::Value v) {
    auto it = idMap.find(v);
    if (it != idMap.end())
      return it->second;
    idMap[v] = "v" + std::to_string(idMap.size());
    return idMap[v];
  }

  std::string getBlockLabel(mlir::Block *block) {
    auto it = blockLabels.find(block);
    if (it != blockLabels.end())
      return it->second;
    blockLabels[block] = "bb" + std::to_string(blockLabels.size());
    return blockLabels[block];
  }

  json brilGenOp(mlir::Operation &op) {
    if (DEBUG)
      llvm::errs() << "entering function brilGenOp\n";

    if (auto constOp = dyn_cast<ConstantOp>(op)) {
      json instrJson;
      instrJson["op"] = "const";
      instrJson["dest"] = getId(constOp.getResult());
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
        if (intAttr.getType().isInteger(64)) {
          instrJson["type"] = "int";
          instrJson["value"] = intAttr.getInt();
        } else if (intAttr.getType().isInteger(1)) {
          instrJson["type"] = "bool";
          instrJson["value"] = static_cast<bool>(intAttr.getInt());
        } else {
          llvm::errs() << "Unsupported constant type in brilGenOp: "
                       << intAttr.getType() << "\n";
          abort();
        }
      }
      return instrJson;
    } else if (auto undefOp = dyn_cast<UndefOp>(op)) {
      json instrJson;
      instrJson["op"] = "undef";
      instrJson["dest"] = getId(undefOp.getResult());
      instrJson["type"] = getTypeJson(undefOp.getResult().getType());
      return instrJson;
    } else if (isa<AddOp>(op) || isa<SubOp>(op) || isa<MulOp>(op) ||
               isa<DivOp>(op) || isa<EqOp>(op) || isa<LtOp>(op) ||
               isa<GtOp>(op) || isa<LeOp>(op) || isa<GeOp>(op) ||
               isa<AndOp>(op) || isa<OrOp>(op)) {
      json instrJson;
      if (isa<AddOp>(op))
        instrJson["op"] = "add";
      else if (isa<SubOp>(op))
        instrJson["op"] = "sub";
      else if (isa<MulOp>(op))
        instrJson["op"] = "mul";
      else if (isa<DivOp>(op))
        instrJson["op"] = "div";
      else if (isa<EqOp>(op))
        instrJson["op"] = "eq";
      else if (isa<LtOp>(op))
        instrJson["op"] = "lt";
      else if (isa<GtOp>(op))
        instrJson["op"] = "gt";
      else if (isa<LeOp>(op))
        instrJson["op"] = "le";
      else if (isa<GeOp>(op))
        instrJson["op"] = "ge";
      else if (isa<AndOp>(op))
        instrJson["op"] = "and";
      else if (isa<OrOp>(op))
        instrJson["op"] = "or";

      instrJson["dest"] = getId(op.getResult(0));
      instrJson["args"] = nlohmann::json::array();
      for (auto operand : op.getOperands()) {
        instrJson["args"].push_back(getId(operand));
      }
      instrJson["type"] = getTypeJson(op.getResult(0).getType());

      return instrJson;
    } else if (auto idOp = dyn_cast<IdOp>(op)) {
      json instrJson;
      instrJson["op"] = "id";
      instrJson["dest"] = getId(idOp.getResult());
      instrJson["args"] = nlohmann::json::array();
      instrJson["args"].push_back(getId(idOp.getInput()));
      instrJson["type"] = getTypeJson(idOp.getResult().getType());
      return instrJson;
    } else if (auto notOp = dyn_cast<NotOp>(op)) {
      json instrJson;
      instrJson["op"] = "not";
      instrJson["dest"] = getId(notOp.getResult());
      instrJson["args"] = nlohmann::json::array();
      instrJson["args"].push_back(getId(notOp->getOperand(0)));
      instrJson["type"] = getTypeJson(notOp.getResult().getType());
      return instrJson;
    } else if (auto callOp = dyn_cast<CallOp>(op)) {
      json instrJson;
      instrJson["op"] = "call";
      instrJson["funcs"] = nlohmann::json::array();
      instrJson["funcs"].push_back(callOp.getCallee().str());
      instrJson["args"] = nlohmann::json::array();
      for (auto operand : callOp.getInputs()) {
        instrJson["args"].push_back(getId(operand));
      }
      if (!callOp.getResults().empty()) {
        instrJson["dest"] = getId(callOp.getResult(0));
        instrJson["type"] = getTypeJson(callOp.getResult(0).getType());
      }
      return instrJson;
    } else if (auto brOp = dyn_cast<BrOp>(op)) {
      json instrArray = nlohmann::json::array();

      auto trueBlock = brOp.getTrueTarget();
      auto falseBlock = brOp.getFalseTarget();

      for (auto entry :
           llvm::zip(trueBlock->getArguments(), brOp.getTrueArgs())) {
        auto arg = std::get<0>(entry);
        auto value = std::get<1>(entry);
        // insert set operation
        json setInstrJson;
        setInstrJson["op"] = "set";
        setInstrJson["args"] = nlohmann::json::array();
        setInstrJson["args"].push_back(getId(arg));
        setInstrJson["args"].push_back(getId(value));

        instrArray.push_back(setInstrJson);
      }

      for (auto entry :
           llvm::zip(falseBlock->getArguments(), brOp.getFalseArgs())) {
        auto arg = std::get<0>(entry);
        auto value = std::get<1>(entry);
        // insert set operation
        json setInstrJson;
        setInstrJson["op"] = "set";
        setInstrJson["args"] = nlohmann::json::array();
        setInstrJson["args"].push_back(getId(arg));
        setInstrJson["args"].push_back(getId(value));

        instrArray.push_back(setInstrJson);
      }

      json brInstrJson;
      brInstrJson["op"] = "br";
      brInstrJson["args"] = nlohmann::json::array();
      brInstrJson["args"].push_back(getId(brOp.getCondition()));
      brInstrJson["labels"] = nlohmann::json::array();
      brInstrJson["labels"].push_back(getBlockLabel(brOp.getTrueTarget()));
      brInstrJson["labels"].push_back(getBlockLabel(brOp.getFalseTarget()));

      instrArray.push_back(brInstrJson);

      return instrArray;
    } else if (auto jmpOp = dyn_cast<JmpOp>(op)) {
      json instrArray = nlohmann::json::array();

      for (auto entry :
           llvm::zip(jmpOp.getTarget()->getArguments(), jmpOp.getArgs())) {
        auto arg = std::get<0>(entry);
        auto value = std::get<1>(entry);
        // insert set operation
        json setInstrJson;
        setInstrJson["op"] = "set";
        setInstrJson["args"] = nlohmann::json::array();
        setInstrJson["args"].push_back(getId(arg));
        setInstrJson["args"].push_back(getId(value));

        instrArray.push_back(setInstrJson);
      }

      json jmpInstrJson;
      jmpInstrJson["op"] = "jmp";
      jmpInstrJson["labels"] = nlohmann::json::array();
      jmpInstrJson["labels"].push_back(getBlockLabel(jmpOp.getTarget()));

      instrArray.push_back(jmpInstrJson);
      return instrArray;
    } else if (auto retOp = dyn_cast<RetOp>(op)) {
      json instrJson;
      instrJson["op"] = "ret";
      if (retOp.getReturnValue()) {
        instrJson["args"] = nlohmann::json::array();
        instrJson["args"].push_back(getId(retOp.getReturnValue()));
      }
      return instrJson;
    } else if (auto printOp = dyn_cast<PrintOp>(op)) {
      json instrJson;
      instrJson["op"] = "print";
      instrJson["args"] = nlohmann::json::array();
      for (auto operand : printOp.getValues()) {
        instrJson["args"].push_back(getId(operand));
      }
      return instrJson;
    } else if (auto nopOp = dyn_cast<NopOp>(op)) {
      json instrJson;
      instrJson["op"] = "nop";
      return instrJson;
    } else if (auto allocOp = dyn_cast<AllocOp>(op)) {
      json instrJson;
      instrJson["op"] = "alloc";
      instrJson["dest"] = getId(allocOp.getResult());
      instrJson["type"] = getTypeJson(allocOp.getResult().getType());
      instrJson["args"] = nlohmann::json::array();
      instrJson["args"].push_back(getId(allocOp.getSize()));
      return instrJson;
    } else if (auto freeOp = dyn_cast<FreeOp>(op)) {
      json instrJson;
      instrJson["op"] = "free";
      instrJson["args"] = nlohmann::json::array();
      instrJson["args"].push_back(getId(freeOp.getPtr()));
      return instrJson;
    } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
      json instrJson;
      instrJson["op"] = "load";
      instrJson["dest"] = getId(loadOp.getResult());
      instrJson["type"] = getTypeJson(loadOp.getResult().getType());
      instrJson["args"] = nlohmann::json::array();
      instrJson["args"].push_back(getId(loadOp.getPtr()));
      return instrJson;
    } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
      json instrJson;
      instrJson["op"] = "store";
      instrJson["args"] = nlohmann::json::array();
      instrJson["args"].push_back(getId(storeOp.getPtr()));
      instrJson["args"].push_back(getId(storeOp.getValue()));
      return instrJson;
    } else if (auto ptrAddOp = dyn_cast<PtrAddOp>(op)) {
      json instrJson;
      instrJson["op"] = "ptradd";
      instrJson["dest"] = getId(ptrAddOp.getResult());
      instrJson["type"] = getTypeJson(ptrAddOp.getResult().getType());
      instrJson["args"] = nlohmann::json::array();
      instrJson["args"].push_back(getId(ptrAddOp.getPtr()));
      instrJson["args"].push_back(getId(ptrAddOp.getOffset()));
      return instrJson;
    }

    else {
      llvm::errs() << "Unsupported operation in brilGenOp: "
                   << op.getName().getStringRef() << "\n";
    }
    abort();
    return {};
  }

  json brilGenBlock(mlir::Block &block, bool entryBlock = false) {
    if (DEBUG)
      llvm::errs() << "entering function brilGenBlock\n";

    json blockJson = nlohmann::json::array();

    blockJson.push_back({{"label", getBlockLabel(&block)}});

    if (!entryBlock) {
      for (auto arg : block.getArguments()) {
        json getJson = {{"dest", getId(arg)},
                        {"op", "get"},
                        {"type", getTypeJson(arg.getType())}};
        blockJson.push_back(getJson);
      }
    }

    for (auto &op : block.getOperations()) {
      auto opJson = brilGenOp(op);
      if (opJson.is_array()) {
        for (auto &&instrJson : opJson) {
          blockJson.push_back(instrJson);
        }
      } else
        blockJson.push_back(opJson);
    }

    return blockJson;
  }

  json brilGenFunc(mlir::bril::FuncOp func) {
    if (DEBUG)
      llvm::errs() << "entering function brilGenFunc "
                   << func.getSymName().str() << "\n";

    idMap.clear();
    blockLabels.clear();

    json funcJson;
    funcJson["name"] = func.getSymName().str();
    funcJson["args"] = nlohmann::json::array();

    for (auto arg : func.getArguments()) {
      json argJson;
      argJson["name"] = getId(arg);
      argJson["type"] = getTypeJson(arg.getType());

      funcJson["args"].push_back(argJson);
    }

    if (!func.getFunctionType().getResults().empty()) {
      auto retType = func.getFunctionType().getResult(0);
      funcJson["type"] = getTypeJson(retType);
    }

    for (auto &block : func.getBlocks()) {
      // sequentially number the blocks
      getBlockLabel(&block);
    }

    bool entryBlock = true;
    for (auto &block : func.getBlocks()) {
      auto blockJson = brilGenBlock(block, entryBlock);
      for (auto &&instrJson : blockJson) {
        funcJson["instrs"].push_back(std::move(instrJson));
      }
      entryBlock = false;
    }

    return funcJson;
  }
};

} // namespace

namespace bril {
nlohmann::json mlirToBril(mlir::ModuleOp module) {
  return MLIR2BrilImpl().brilGenModule(module);
}
} // namespace bril