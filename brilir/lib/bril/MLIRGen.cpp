//===- MLIRGen.cpp - MLIR Generation from a Bril JSON
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Bril JSON
// for the Bril language.
//
//===----------------------------------------------------------------------===//

#include "bril/MLIRGen.h"
#include "bril/BrilOps.h"
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
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <nlohmann/json_fwd.hpp>
#include <unordered_map>
#include <vector>

using namespace mlir::bril;
using namespace bril;

using llvm::SmallVector;
using llvm::StringRef;

namespace {

/// Implementation of a simple MLIR emission from the Bril JSON.
///
/// This will emit operations that are specific to the Bril language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {
    DEBUG = getenv("DEBUG") != nullptr;
  }

  mlir::OwningOpRef<mlir::ModuleOp> mlirGen(nlohmann::json &json) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGen\n";
    // Create the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto &funcJson : json["functions"]) {
      labelToBlock.clear();
      blockList.clear();
      blockToLabel.clear();
      symbolTable.clear();
      if (llvm::failed(mlirGenFunction(funcJson))) {
        theModule->emitError("failed to generate function");
        return nullptr;
      }
    }

    if (llvm::failed(mlir::verify(theModule))) {
      theModule->emitError("module verification failed");
      return nullptr;
    }

    return theModule;
  }

private:
  struct BlockInfo {
    mlir::Block *block;
    llvm::SmallVector<std::string, 4> blockArgs;
    llvm::StringMap<mlir::Value> ssaSets;
  };

  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  std::unordered_map<std::string, mlir::Value> symbolTable;
  std::unordered_map<std::string, BlockInfo> labelToBlock;
  std::unordered_map<mlir::Block *, std::string> blockToLabel;
  std::vector<mlir::Block *> blockList;
  bool DEBUG;

  llvm::LogicalResult declare(std::string var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable[var] = value;
    return mlir::success();
  }

  mlir::Type getType(const nlohmann::json &type) {
    if (type == "int")
      return builder.getIntegerType(64);
    if (type == "bool")
      return builder.getIntegerType(1);
    if (type["ptr"] == "int")
      return mlir::bril::PtrType::get(builder.getContext(),
                                      builder.getIntegerType(64));
    if (type["ptr"] == "bool")
      return mlir::bril::PtrType::get(builder.getContext(),
                                      builder.getIntegerType(1));
    return nullptr;
  }

  std::string generateBlockName() {
    static int blockCounter = 0;
    return "___generated_block_" + std::to_string(blockCounter++);
  }

  std::vector<std::vector<nlohmann::json>>
  splitBlocks(nlohmann::json &instrsJson) {
    std::vector<std::vector<nlohmann::json>> blocks = {};
    std::vector<nlohmann::json> currentBlock = {};

    for (auto &instrJson : instrsJson) {
      if (instrJson.contains("op")) {
        currentBlock.push_back(instrJson);

        auto op = instrJson["op"].get<std::string>();

        if (op == "br" || op == "jmp" || op == "ret") {
          blocks.push_back(currentBlock);
          currentBlock = {};
        }
      } else {
        if (!currentBlock.empty()) {
          blocks.push_back(currentBlock);
        }

        currentBlock = {instrJson};
      }
    }

    if (!currentBlock.empty()) {
      blocks.push_back(currentBlock);
    }

    return blocks;
  }

  llvm::LogicalResult mlirGenFunction(nlohmann::json &funcJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenFunction "
                   << funcJson["name"].get<std::string>() << "\n";

    auto funcName = funcJson["name"].get<std::string>();

    mlir::SmallVector<mlir::Type, 4> argTypes = {};
    for (auto &arg : funcJson["args"]) {
      auto argType = getType(arg["type"]);
      argTypes.push_back(argType);
    }

    mlir::TypeRange returnTypes = {};
    if (funcJson.contains("type")) {
      returnTypes = {getType(funcJson["type"])};
    }

    builder.setInsertionPointToEnd(theModule.getBody());

    auto func = FuncOp::create(builder, builder.getUnknownLoc(), funcName,
                               builder.getFunctionType(argTypes, returnTypes));

    auto &entryBlock = func.front();
    for (auto nameValue :
         llvm::zip(funcJson["args"], entryBlock.getArguments())) {
      auto name = std::get<0>(nameValue)["name"].get<std::string>();
      auto value = std::get<1>(nameValue);

      if (llvm::failed(declare(name, value))) {
        func.emitError("failed to declare argument ") << name;
        return llvm::failure();
      }
    }

    builder.setInsertionPointToEnd(&entryBlock);

    auto blocks = splitBlocks(funcJson["instrs"]);

    bool firstBlock = true;
    for (auto &block : blocks) {
      if (block.front().contains("label")) {
        auto labelName = block.front()["label"].get<std::string>();
        auto *mlirBlock = firstBlock ? &entryBlock : func.addBlock();
        llvm::SmallVector<std::string, 4> blockArgNames;

        for (auto &instr : block) {
          // collect all block arguments from 'get' instructions
          if (instr.contains("op") && instr["op"] == "get") {
            auto blockArg = mlirBlock->addArgument(getType(instr["type"]),
                                                   builder.getUnknownLoc());
            blockArgNames.push_back(instr["dest"].get<std::string>());
            if (llvm::failed(
                    declare(instr["dest"].get<std::string>(), blockArg))) {
              func.emitError("failed to declare block argument ");
              return llvm::failure();
            }
          }
        }

        labelToBlock[labelName] = BlockInfo{mlirBlock, blockArgNames, {}};
        blockToLabel[mlirBlock] = labelName;

        blockList.push_back(mlirBlock);
      } else {
        auto blockName = generateBlockName();
        auto *mlirBlock = firstBlock ? &entryBlock : func.addBlock();
        llvm::SmallVector<std::string, 4> blockArgNames;

        for (auto &instr : block) {
          // collect all block arguments from 'get' instructions
          if (instr.contains("op") && instr["op"] == "get") {
            auto blockArg = mlirBlock->addArgument(getType(instr["type"]),
                                                   builder.getUnknownLoc());
            blockArgNames.push_back(instr["dest"].get<std::string>());
            if (llvm::failed(
                    declare(instr["dest"].get<std::string>(), blockArg))) {
              func.emitError("failed to declare block argument ");
              return llvm::failure();
            }
          }
        }

        labelToBlock[blockName] = BlockInfo{mlirBlock, {}, {}};
        blockToLabel[mlirBlock] = blockName;

        block.insert(block.begin(),
                     nlohmann::json{{"label", blockName}}); // add a label

        blockList.push_back(mlirBlock);
      }
      firstBlock = false;
    }

    for (auto [blockIdx, block] : llvm::enumerate(blocks)) {
      BlockInfo *blockInfo = nullptr;
      if (block.front().contains("label")) {
        auto labelName = block.front()["label"].get<std::string>();
        if (!labelToBlock.count(labelName)) {
          llvm::errs() << "Undefined label: " << labelName << "\n";
          return llvm::failure();
        }
        blockInfo = &labelToBlock[labelName];
      }
      builder.setInsertionPointToEnd(blockList[blockIdx]);
      for (auto instr : block) {
        if (llvm::failed(mlirGenInstruction(instr, blockInfo))) {
          func.emitError("failed to generate instruction");
          return llvm::failure();
        }
      }

      if ((block.back().contains("op") && block.back()["op"] != "br" &&
           block.back()["op"] != "jmp" && block.back()["op"] != "ret") ||
          block.back().contains("label")) {
        // create jmp to the next block if it exists
        if (blockIdx + 1 < blockList.size()) {
          nlohmann::json jmpJson = {
              {"labels", {blockToLabel[blockList[blockIdx + 1]]}},
              {"op", "jmp"}};
          if (llvm::failed(mlirGenJmp(jmpJson, blockInfo))) {
            func.emitError("failed to generate jmp to next block");
            return llvm::failure();
          }
        } else {
          // otherwise just generate a dummy ret
          if (!funcJson.contains("type") || funcJson["type"].contains("ptr")) {
            RetOp::create(builder, builder.getUnknownLoc(), mlir::Value{});
          } else {
            UndefOp undefOp = UndefOp::create(builder, builder.getUnknownLoc(),
                                              getType(funcJson["type"]));
            RetOp::create(builder, builder.getUnknownLoc(),
                          undefOp.getResult());
          }
        }
      }
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenInstruction(nlohmann::json &instrJson,
                                         BlockInfo *blockInfo) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenInstruction "
                   << instrJson.dump() << " " << blockInfo << "\n";
    if (!instrJson.contains("op")) {
      // label instruction, already handled in block generation
      return llvm::success();
    }

    auto op = instrJson["op"].get<std::string>();

    if (op == "get") {
      // already handled in block generation
      return llvm::success();
    } else if (op == "const") {
      return mlirGenConst(instrJson);
    } else if (op == "add" || op == "sub" || op == "mul" || op == "div" ||
               op == "eq" || op == "lt" || op == "gt" || op == "le" ||
               op == "ge" || op == "and" || op == "or") {
      return mlirGenBinaryOp(instrJson);
    } else if (op == "undef") {
      return mlirGenUndef(instrJson);
    } else if (op == "id") {
      return mlirGenId(instrJson);
    } else if (op == "not") {
      return mlirGenNot(instrJson);
    } else if (op == "br") {
      return mlirGenBranch(instrJson, blockInfo);
    } else if (op == "jmp") {
      return mlirGenJmp(instrJson, blockInfo);
    } else if (op == "ret") {
      return mlirGenRet(instrJson);
    } else if (op == "set") {
      return mlirGenSet(instrJson, blockInfo);
    } else if (op == "print") {
      return mlirGenPrint(instrJson);
    } else if (op == "nop") {
      return mlirGenNop(instrJson);
    } else if (op == "call") {
      return mlirGenCall(instrJson);
    } else if (op == "alloc") {
      return mlirGenAlloc(instrJson);
    } else if (op == "load") {
      return mlirGenLoad(instrJson);
    } else if (op == "store") {
      return mlirGenStore(instrJson);
    } else if (op == "free") {
      return mlirGenFree(instrJson);
    } else if (op == "ptradd") {
      return mlirGenPtrAdd(instrJson);
    } else {
      llvm::errs() << "Unhandled operation: " << op << "\n";
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenConst(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenConst " << instrJson.dump()
                   << "\n";
    auto dest = instrJson["dest"].get<std::string>();
    if (instrJson["type"] == "int") {
      int64_t value = instrJson["value"].get<int64_t>();
      auto constOp =
          ConstantOp::create(builder, builder.getUnknownLoc(), value);
      if (llvm::failed(declare(dest, constOp.getResult()))) {
        return mlir::failure();
      }
    } else if (instrJson["type"] == "bool") {
      bool value = instrJson["value"].get<bool>();
      auto constOp =
          ConstantOp::create(builder, builder.getUnknownLoc(), value);
      if (llvm::failed(declare(dest, constOp.getResult()))) {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }
    return llvm::success();
  }

  llvm::LogicalResult mlirGenUndef(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenUndef " << instrJson.dump()
                   << "\n";
    auto dest = instrJson["dest"].get<std::string>();
    auto undefOp = UndefOp::create(builder, builder.getUnknownLoc(),
                                   getType(instrJson["type"]));
    if (llvm::failed(declare(dest, undefOp.getResult()))) {
      return mlir::failure();
    }
    return llvm::success();
  }

  llvm::LogicalResult mlirGenId(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenId " << instrJson.dump()
                   << "\n";

    auto dest = instrJson["dest"].get<std::string>();
    auto argName = instrJson["args"][0].get<std::string>();

    if (!symbolTable.count(argName)) {
      llvm::errs() << "Undefined variable in id operation: " << argName << "\n";
      return mlir::failure();
    }

    auto arg = symbolTable[argName];
    auto idOp = IdOp::create(builder, builder.getUnknownLoc(),
                             getType(instrJson["type"]), arg);

    if (llvm::failed(declare(dest, idOp.getResult()))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenNot(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenNot " << instrJson.dump()
                   << "\n";
    auto dest = instrJson["dest"].get<std::string>();
    auto argName = instrJson["args"][0].get<std::string>();

    if (!symbolTable.count(argName)) {
      llvm::errs() << "Undefined variable in not operation: " << argName
                   << "\n";
      return mlir::failure();
    }

    auto arg = symbolTable[argName];

    auto notOp = NotOp::create(builder, builder.getUnknownLoc(), arg);
    auto result = notOp.getResult();

    if (llvm::failed(declare(dest, result))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenBranch(nlohmann::json &instrJson,
                                    BlockInfo *blockInfo) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenBranch " << instrJson.dump()
                   << " " << blockInfo << "\n";
    if (!blockInfo) {
      llvm::errs() << "Branch operation on a block without BlockInfo\n";
      return mlir::failure();
    }

    auto argName = instrJson["args"][0].get<std::string>();

    if (!symbolTable.count(argName)) {
      llvm::errs() << "Undefined variable in branch operation: " << argName
                   << "\n";
      return mlir::failure();
    }

    auto arg = symbolTable[argName];

    auto trueLabel = instrJson["labels"][0].get<std::string>();
    auto falseLabel = instrJson["labels"][1].get<std::string>();

    if (!labelToBlock.count(trueLabel) || !labelToBlock.count(falseLabel)) {
      llvm::errs() << "Undefined label in branch operation: " << trueLabel
                   << " or " << falseLabel << "\n";
      return mlir::failure();
    }

    auto trueBlock = labelToBlock[trueLabel];
    auto falseBlock = labelToBlock[falseLabel];

    llvm::SmallVector<mlir::Value, 4> trueArgs = {};
    llvm::SmallVector<mlir::Value, 4> falseArgs = {};

    for (auto &arg : trueBlock.blockArgs) {
      if (!blockInfo->ssaSets.count(arg)) {
        llvm::errs() << "Undefined variable in branch true args: " << arg
                     << "\n";
        return mlir::failure();
      }
      trueArgs.push_back(blockInfo->ssaSets.lookup(arg));
    }

    for (auto &arg : falseBlock.blockArgs) {
      if (!blockInfo->ssaSets.count(arg)) {
        llvm::errs() << "Undefined variable in branch false args: " << arg
                     << "\n";
        return mlir::failure();
      }
      falseArgs.push_back(blockInfo->ssaSets.lookup(arg));
    }

    BrOp::create(builder, builder.getUnknownLoc(), arg, trueArgs, falseArgs,
                 trueBlock.block, falseBlock.block);

    return llvm::success();
  }

  llvm::LogicalResult mlirGenCall(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenCall " << instrJson.dump()
                   << "\n";
    auto funcName = instrJson["funcs"][0].get<std::string>();

    auto args = instrJson["args"];

    SmallVector<mlir::Value, 4> mlirArgs = {};
    for (auto &argNameJson : args) {
      auto argName = argNameJson.get<std::string>();
      if (!symbolTable.count(argName)) {
        llvm::errs() << "Undefined variable in call operation: " << argName
                     << "\n";
        return mlir::failure();
      }
      mlirArgs.push_back(symbolTable[argName]);
    }

    if (instrJson.contains("dest")) {
      auto dest = instrJson["dest"].get<std::string>();
      auto type = getType(instrJson["type"]);
      auto callOp = CallOp::create(
          builder, builder.getUnknownLoc(), type,
          mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName),
          mlirArgs, mlir::ArrayAttr(), mlir::ArrayAttr());

      if (llvm::failed(declare(dest, callOp.getResult(0)))) {
        llvm::errs() << "Failed to declare variable: " << dest << "\n";
        return mlir::failure();
      }
    } else {
      CallOp::create(
          builder, builder.getUnknownLoc(), mlir::TypeRange{},
          mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName),
          mlirArgs, mlir::ArrayAttr(), mlir::ArrayAttr());
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenPrint(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenPrint " << instrJson.dump()
                   << "\n";
    SmallVector<mlir::Value, 4> args = {};

    for (auto &argNameJson : instrJson["args"]) {
      auto argName = argNameJson.get<std::string>();

      if (!symbolTable.count(argName)) {
        llvm::errs() << "Undefined variable in print operation: " << argName
                     << "\n";
        return mlir::failure();
      }

      args.push_back(symbolTable[argName]);
    }

    PrintOp::create(builder, builder.getUnknownLoc(), args);

    return llvm::success();
  }

  llvm::LogicalResult mlirGenNop(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenNop " << instrJson.dump()
                   << "\n";
    NopOp::create(builder, builder.getUnknownLoc());
    return llvm::success();
  }

  llvm::LogicalResult mlirGenRet(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenRet " << instrJson.dump()
                   << "\n";
    if (instrJson.contains("args") && !instrJson["args"].empty()) {
      auto argName = instrJson["args"][0].get<std::string>();

      if (!symbolTable.count(argName)) {
        llvm::errs() << "Undefined variable in ret operation: " << argName
                     << "\n";
        return mlir::failure();
      }

      auto arg = symbolTable[argName];

      RetOp::create(builder, builder.getUnknownLoc(), arg);
    } else {
      RetOp::create(builder, builder.getUnknownLoc(), mlir::Value{});
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenJmp(nlohmann::json &instrJson,
                                 BlockInfo *blockInfo) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenJmp " << instrJson.dump() << " "
                   << blockInfo << "\n";

    auto targetLabel = instrJson["labels"][0].get<std::string>();

    if (!labelToBlock.count(targetLabel)) {
      llvm::errs() << "Undefined label in jmp operation: " << targetLabel
                   << "\n";
      return mlir::failure();
    }

    auto targetBlock = labelToBlock[targetLabel];

    llvm::SmallVector<mlir::Value, 4> args = {};
    if (blockInfo) {
      for (auto &arg : targetBlock.blockArgs) {
        if (!blockInfo->ssaSets.count(arg)) {
          llvm::errs() << "Undefined variable in jmp args: " << arg << "\n";
          return mlir::failure();
        }
        args.push_back(blockInfo->ssaSets.lookup(arg));
      }
    }

    JmpOp::create(builder, builder.getUnknownLoc(), args, targetBlock.block);

    return llvm::success();
  }

  llvm::LogicalResult mlirGenBinaryOp(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenBinaryOp " << instrJson.dump()
                   << "\n";
    auto dest = instrJson["dest"].get<std::string>();
    auto arg1Name = instrJson["args"][0].get<std::string>();
    auto arg2Name = instrJson["args"][1].get<std::string>();

    if (!symbolTable.count(arg1Name) || !symbolTable.count(arg2Name)) {
      llvm::errs() << "Undefined variable in binary operation: " << arg1Name
                   << " or " << arg2Name << "\n";
      return mlir::failure();
    }

    auto arg1 = symbolTable[arg1Name];
    auto arg2 = symbolTable[arg2Name];

    mlir::Value result;

    if (instrJson["op"] == "add") {
      auto addOp = AddOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = addOp.getResult();
    } else if (instrJson["op"] == "sub") {
      auto subOp = SubOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = subOp.getResult();
    } else if (instrJson["op"] == "mul") {
      auto mulOp = MulOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = mulOp.getResult();
    } else if (instrJson["op"] == "div") {
      auto divOp = DivOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = divOp.getResult();
    } else if (instrJson["op"] == "eq") {
      auto eqOp = EqOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = eqOp.getResult();
    } else if (instrJson["op"] == "lt") {
      auto ltOp = LtOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = ltOp.getResult();
    } else if (instrJson["op"] == "gt") {
      auto gtOp = GtOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = gtOp.getResult();
    } else if (instrJson["op"] == "le") {
      auto leOp = LeOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = leOp.getResult();
    } else if (instrJson["op"] == "ge") {
      auto geOp = GeOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = geOp.getResult();
    } else if (instrJson["op"] == "and") {
      auto andOp = AndOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = andOp.getResult();
    } else if (instrJson["op"] == "or") {
      auto orOp = OrOp::create(builder, builder.getUnknownLoc(), arg1, arg2);
      result = orOp.getResult();
    }

    if (llvm::failed(declare(dest, result))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenSet(nlohmann::json &instrJson,
                                 BlockInfo *blockInfo) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenSet " << instrJson.dump() << " "
                   << blockInfo << "\n";
    if (!blockInfo) {
      llvm::errs() << "Set operation on a block without BlockInfo\n";
      return mlir::failure();
    }

    auto dest = instrJson["args"][0].get<std::string>();
    auto src = instrJson["args"][1].get<std::string>();

    mlir::Value arg;

    if (!symbolTable.count(src)) {
      auto destType = symbolTable[dest].getType();
      nlohmann::json undefJson = {
          {"dest", "___undef__" + src},
          {"type", destType.isInteger(1) ? "bool" : "int"},
          {"op", "undef"}};
      mlirGenUndef(undefJson);
      arg = symbolTable["___undef__" + src];
    } else {
      arg = symbolTable[src];
    }

    blockInfo->ssaSets[dest] = arg;

    return llvm::success();
  }

  llvm::LogicalResult mlirGenAlloc(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenAlloc " << instrJson.dump()
                   << "\n";

    auto dest = instrJson["dest"].get<std::string>();
    auto sizeName = instrJson["args"][0].get<std::string>();

    if (!symbolTable.count(sizeName)) {
      llvm::errs() << "Undefined variable in alloc operation: " << sizeName
                   << "\n";
      return mlir::failure();
    }
    auto size = symbolTable[sizeName];

    auto type = getType(instrJson["type"]);

    auto allocOp =
        AllocOp::create(builder, builder.getUnknownLoc(), type, size);

    if (llvm::failed(declare(dest, allocOp.getResult()))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenFree(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenFree " << instrJson.dump()
                   << "\n";

    auto ptrName = instrJson["args"][0].get<std::string>();

    if (!symbolTable.count(ptrName)) {
      llvm::errs() << "Undefined variable in free operation: " << ptrName
                   << "\n";
      return mlir::failure();
    }
    auto ptr = symbolTable[ptrName];

    FreeOp::create(builder, builder.getUnknownLoc(), ptr);

    return llvm::success();
  }

  llvm::LogicalResult mlirGenLoad(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenLoad " << instrJson.dump()
                   << "\n";

    auto dest = instrJson["dest"].get<std::string>();
    auto ptrName = instrJson["args"][0].get<std::string>();
    auto type = getType(instrJson["type"]);

    if (!symbolTable.count(ptrName)) {
      llvm::errs() << "Undefined variable in load operation: " << ptrName
                   << "\n";
      return mlir::failure();
    }
    auto ptr = symbolTable[ptrName];

    auto loadOp = LoadOp::create(builder, builder.getUnknownLoc(), type, ptr);

    if (llvm::failed(declare(dest, loadOp.getResult()))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }

  llvm::LogicalResult mlirGenStore(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenStore " << instrJson.dump()
                   << "\n";

    auto ptrName = instrJson["args"][0].get<std::string>();
    auto valueName = instrJson["args"][1].get<std::string>();

    if (!symbolTable.count(ptrName)) {
      llvm::errs() << "Undefined variable in store operation: " << ptrName
                   << "\n";
      return mlir::failure();
    }
    auto ptr = symbolTable[ptrName];

    if (!symbolTable.count(valueName)) {
      llvm::errs() << "Undefined variable in store operation: " << valueName
                   << "\n";
      return mlir::failure();
    }
    auto value = symbolTable[valueName];

    StoreOp::create(builder, builder.getUnknownLoc(), ptr, value);

    return llvm::success();
  }

  llvm::LogicalResult mlirGenPtrAdd(nlohmann::json &instrJson) {
    if (DEBUG)
      llvm::errs() << "entering function mlirGenPtrAdd " << instrJson.dump()
                   << "\n";

    auto dest = instrJson["dest"].get<std::string>();
    auto ptrName = instrJson["args"][0].get<std::string>();
    auto offsetName = instrJson["args"][1].get<std::string>();
    auto type = getType(instrJson["type"]);

    if (!symbolTable.count(ptrName)) {
      llvm::errs() << "Undefined variable in ptradd operation: " << ptrName
                   << "\n";
      return mlir::failure();
    }
    auto ptr = symbolTable[ptrName];

    if (!symbolTable.count(offsetName)) {
      llvm::errs() << "Undefined variable in ptradd operation: " << offsetName
                   << "\n";
      return mlir::failure();
    }
    auto offset = symbolTable[offsetName];

    auto ptrAddOp =
        PtrAddOp::create(builder, builder.getUnknownLoc(), type, ptr, offset);

    if (llvm::failed(declare(dest, ptrAddOp.getResult()))) {
      llvm::errs() << "Failed to declare variable: " << dest << "\n";
      return mlir::failure();
    }

    return llvm::success();
  }
};

} // namespace

namespace bril {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          nlohmann::json &json) {
  return MLIRGenImpl(context).mlirGen(json);
}

} // namespace bril
