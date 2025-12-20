//===- BrilPasses.cpp - Bril passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "bril/BrilOps.h"
#include "bril/BrilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "bril/BrilPasses.h"

using llvm::cast, llvm::dyn_cast;

namespace mlir::bril {
#define GEN_PASS_DEF_CONVERTBRILTOSTD
#include "bril/BrilPasses.h.inc"

#define GEN_PASS_DEF_RENAMEMAINFUNCTION
#include "bril/BrilPasses.h.inc"

namespace {
struct ConstantOpConversion : public OpConversionPattern<bril::ConstantOp> {
  using OpConversionPattern<bril::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, intAttr);
    } else {
      return failure();
    }

    return success();
  }
};

struct IdOpConversion : public OpConversionPattern<bril::IdOp> {
  using OpConversionPattern<bril::IdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::IdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getResult().getType().isInteger()) {
      auto const0 = arith::ConstantIntOp::create(
          rewriter, op.getLoc(), 0,
          op.getResult().getType().isInteger(64) ? 64 : 1);
      auto add0 =
          arith::AddIOp::create(rewriter, op.getLoc(), op.getInput(), const0);
      rewriter.replaceOp(op, add0.getResult());
    } else {
      auto bitcastOp = LLVM::BitcastOp::create(
          rewriter, op.getLoc(),
          LLVM::LLVMPointerType::get(rewriter.getContext()), adaptor.getInput());
      rewriter.replaceOp(op, bitcastOp.getResult());
    }
    return success();
  }
};

struct UndefOpConversion : public OpConversionPattern<bril::UndefOp> {
  using OpConversionPattern<bril::UndefOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::UndefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getResult().getType().isInteger()) {
      auto zeroAttr = rewriter.getIntegerAttr(op->getResult(0).getType(), 0);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, zeroAttr);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(
          op, LLVM::LLVMPointerType::get(rewriter.getContext()));
    }
    return success();
  }
};

template <typename Op, typename LoweredOp>
struct BinaryOpConversion : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<Op>::OpAdaptor;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto lhs = op.getOperands()[0];
    auto rhs = op.getOperands()[1];

    rewriter.replaceOpWithNewOp<LoweredOp>(op, lhs, rhs);

    return success();
  }
};

template <typename Op, arith::CmpIPredicate Predicate>
struct CmpOpConversion : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<Op>::OpAdaptor;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto lhs = op.getOperands()[0];
    auto rhs = op.getOperands()[1];

    auto loc = op.getLoc();

    auto cmpOp = arith::CmpIOp::create(rewriter, loc, Predicate, lhs, rhs);

    rewriter.replaceOp(op, cmpOp.getResult());

    return success();
  }
};

using AddOpConversion = BinaryOpConversion<bril::AddOp, arith::AddIOp>;
using SubOpConversion = BinaryOpConversion<bril::SubOp, arith::SubIOp>;
using MulOpConversion = BinaryOpConversion<bril::MulOp, arith::MulIOp>;
using DivOpConversion = BinaryOpConversion<bril::DivOp, arith::DivSIOp>;
using EqOpConversion = CmpOpConversion<bril::EqOp, arith::CmpIPredicate::eq>;
using LtOpConversion = CmpOpConversion<bril::LtOp, arith::CmpIPredicate::slt>;
using GtOpConversion = CmpOpConversion<bril::GtOp, arith::CmpIPredicate::sgt>;
using LeOpConversion = CmpOpConversion<bril::LeOp, arith::CmpIPredicate::sle>;
using GeOpConversion = CmpOpConversion<bril::GeOp, arith::CmpIPredicate::sge>;
using AndOpConversion = BinaryOpConversion<bril::AndOp, arith::AndIOp>;
using OrOpConversion = BinaryOpConversion<bril::OrOp, arith::OrIOp>;
// not

struct NotOpConversion : public OpConversionPattern<bril::NotOp> {
  using OpConversionPattern<bril::NotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto input = op.getOperand();
    auto loc = op.getLoc();

    auto const1 = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto xorOp = arith::XOrIOp::create(rewriter, loc, input, const1);

    rewriter.replaceOp(op, xorOp.getResult());

    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<bril::CallOp> {
  using OpConversionPattern<bril::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type> newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes))) {
      op->emitError("Failed to convert result types in CallOp");
      return failure();
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, op.getCallee(), newResultTypes, adaptor.getOperands());
    return success();
  }
};

struct JmpOpConversion : public OpConversionPattern<bril::JmpOp> {
  using OpConversionPattern<bril::JmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::JmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getTarget(),
                                              adaptor.getOperands());
    return success();
  }
};

struct BrOpConversion : public OpConversionPattern<bril::BrOp> {
  using OpConversionPattern<bril::BrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::BrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, op.getCondition(), op.getTrueTarget(), adaptor.getTrueArgs(),
        op.getFalseTarget(), adaptor.getFalseArgs());
    return success();
  }
};

struct FuncOpConversion : public OpConversionPattern<bril::FuncOp> {
  using OpConversionPattern<bril::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    SmallVector<Type> newArgTypes, newResultTypes;
    if (failed(
            typeConverter->convertTypes(op.getArgumentTypes(), newArgTypes)) ||
        failed(
            typeConverter->convertTypes(op.getResultTypes(), newResultTypes))) {
      op->emitError("Failed to convert function argument or result types");
      return failure();
    }

    auto newFuncType = rewriter.getFunctionType(newArgTypes, newResultTypes);

    auto newFuncOp =
        func::FuncOp::create(rewriter, op.getLoc(), op.getName(), newFuncType);
    // Inline the body of the old function into the new function.
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(),
                                           *typeConverter))) {
      op->emitError("Failed to convert function body types");
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct RetOpConversion : public OpConversionPattern<bril::RetOp> {
  using OpConversionPattern<bril::RetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::RetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct NopOpConversion : public OpConversionPattern<bril::NopOp> {
  using OpConversionPattern<bril::NopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::NopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

struct PrintOpConversion : public OpConversionPattern<bril::PrintOp> {
  using OpConversionPattern<bril::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // For now, just erase the print operation.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);

    Location loc = op.getLoc();

    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%ld\0", 4), parentModule);
    Value spaceCst = getOrCreateGlobalString(loc, rewriter, "space",
                                             StringRef(" \0", 2), parentModule);
    Value newlineCst = getOrCreateGlobalString(
        loc, rewriter, "newline", StringRef("\n\0", 2), parentModule);

    const auto numValues = op.getValues().size();
    for (auto [i, val] : llvm::enumerate(op.getValues())) {
      // Call printf with the format specifier and the value to print.
      if (val.getType().isInteger(64)) {
        LLVM::CallOp::create(rewriter, loc,
                             getPrintfType(rewriter.getContext()), printfRef,
                             ArrayRef<Value>({formatSpecifierCst, val}));
      } else if (val.getType().isInteger(1)) {
        Value trueStr = getOrCreateGlobalString(
            loc, rewriter, "true_str", StringRef("true\0", 5), parentModule);
        Value falseStr = getOrCreateGlobalString(
            loc, rewriter, "false_str", StringRef("false\0", 6), parentModule);

        Value boolAsStr =
            arith::SelectOp::create(rewriter, loc, val, trueStr, falseStr);
        LLVM::CallOp::create(rewriter, loc,
                             getPrintfType(rewriter.getContext()), printfRef,
                             ArrayRef<Value>({boolAsStr}));
      } else {
        // Unsupported type for printing.
        return failure();
      }

      // If this is not the last value, print a space after it.
      if (i != numValues - 1) {
        LLVM::CallOp::create(rewriter, loc,
                             getPrintfType(rewriter.getContext()), printfRef,
                             ArrayRef<Value>({spaceCst}));
      }
    }

    // Print a newline at the end.
    LLVM::CallOp::create(rewriter, loc, getPrintfType(rewriter.getContext()),
                         printfRef, ArrayRef<Value>({newlineCst}));

    rewriter.eraseOp(op);
    return success();
  }

private:
  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, llvmPtrTy,
                                                  /*isVarArg=*/true);
    return llvmFnType;
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf",
                             getPrintfType(context));
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/true,
                                      LLVM::Linkage::Internal, name,
                                      builder.getStringAttr(value),
                                      /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
    Value cst0 = LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                          builder.getIndexAttr(0));
    return LLVM::GEPOp::create(
        builder, loc, LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

struct AllocOpConversion : public OpConversionPattern<bril::AllocOp> {
  using OpConversionPattern<bril::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    // replace with malloc
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto mallocRef = getOrInsertMalloc(rewriter, parentModule);

    auto sizeofType = arith::ConstantIntOp::create(
        rewriter, loc,
        (op.getType().getPointeeType().getIntOrFloatBitWidth() + 7) / 8, 64);
    auto sizeInBytes =
        arith::MulIOp::create(rewriter, loc, op.getSize(), sizeofType);

    auto mallocCall = LLVM::CallOp::create(
        rewriter, loc, getMallocType(rewriter.getContext()), mallocRef,
        ArrayRef<Value>({sizeInBytes.getResult()}));

    rewriter.replaceOp(op, mallocCall.getResult());

    return success();
  }

private:
  static LLVM::LLVMFunctionType getMallocType(MLIRContext *context) {
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmPtrTy, {llvmI64Ty});
    return llvmFnType;
  }

  /// Return a symbol reference to the malloc function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertMalloc(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("malloc"))
      return SymbolRefAttr::get(context, "malloc");

    // Insert the malloc function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "malloc",
                             getMallocType(context));
    return SymbolRefAttr::get(context, "malloc");
  }
};

struct FreeOpConversion : public OpConversionPattern<bril::FreeOp> {
  using OpConversionPattern<bril::FreeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::FreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();

    auto parentModule = op->getParentOfType<ModuleOp>();
    auto freeRef = getOrInsertFree(rewriter, parentModule);

    auto ptr = adaptor.getPtr();

    LLVM::CallOp::create(rewriter, loc, getFreeType(rewriter.getContext()),
                         freeRef, ArrayRef<Value>({ptr}));

    rewriter.eraseOp(op);

    return success();
  }

private:
  static LLVM::LLVMFunctionType getFreeType(MLIRContext *context) {
    auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmVoidType = LLVM::LLVMVoidType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrTy});
    return llvmFnType;
  }

  /// Return a symbol reference to the free function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertFree(PatternRewriter &rewriter,
                                           ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("free"))
      return SymbolRefAttr::get(context, "free");

    // Insert the free function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "free",
                             getFreeType(context));
    return SymbolRefAttr::get(context, "free");
  }
};

struct LoadOpConversion : public OpConversionPattern<bril::LoadOp> {
  using OpConversionPattern<bril::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loadOp = mlir::LLVM::LoadOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), adaptor.getPtr());
    rewriter.replaceOp(op, loadOp.getResult());
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<bril::StoreOp> {
  using OpConversionPattern<bril::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(),
                                                     adaptor.getPtr());
    return success();
  }
};

struct PtrAddOpConversion : public OpConversionPattern<bril::PtrAddOp> {
  using OpConversionPattern<bril::PtrAddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bril::PtrAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto getElemPtrOp = mlir::LLVM::GEPOp::create(
        rewriter, op.getLoc(),
        LLVM::LLVMPointerType::get(rewriter.getContext()),
        op.getPtr().getType().getPointeeType(), adaptor.getPtr(),
        ArrayRef<Value>({op.getOffset()}));
    rewriter.replaceOp(op, getElemPtrOp.getResult());
    return success();
  }
};
} // namespace

class ConvertBrilToStd : public impl::ConvertBrilToStdBase<ConvertBrilToStd> {
public:
  using impl::ConvertBrilToStdBase<ConvertBrilToStd>::ConvertBrilToStdBase;
  void runOnOperation() final {
    ConversionTarget target(getContext());

    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                           mlir::cf::ControlFlowDialect,
                           mlir::LLVM::LLVMDialect>();

    target.addIllegalDialect<mlir::bril::BrilDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([&](bril::PtrType type) -> Type {
      auto llvmPtrType = LLVM::LLVMPointerType::get(&getContext());
      return llvmPtrType;
    });

    RewritePatternSet patterns(&getContext());
    patterns
        .add<ConstantOpConversion, IdOpConversion, UndefOpConversion,
             AddOpConversion, SubOpConversion, MulOpConversion, DivOpConversion,
             EqOpConversion, LtOpConversion, GtOpConversion, LeOpConversion,
             GeOpConversion, AndOpConversion, OrOpConversion, NotOpConversion,
             CallOpConversion, JmpOpConversion, BrOpConversion,
             FuncOpConversion, RetOpConversion, NopOpConversion,
             PrintOpConversion, AllocOpConversion, FreeOpConversion,
             LoadOpConversion, StoreOpConversion, PtrAddOpConversion>(
            typeConverter, &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

class RenameMainFunction
    : public impl::RenameMainFunctionBase<RenameMainFunction> {

public:
  using impl::RenameMainFunctionBase<
      RenameMainFunction>::RenameMainFunctionBase;

  void runOnOperation() final {
    ModuleOp op = getOperation();

    SymbolTable symbolTable(op);
    for (auto func : op.getOps<func::FuncOp>()) {
      if (func.getSymName() == "main") {
        if (failed(symbolTable.rename(func, "bril_main"))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};
} // namespace mlir::bril
  // namespace mlir::bril
