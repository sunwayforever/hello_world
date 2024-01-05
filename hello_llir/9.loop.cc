#include <llvm-16/llvm/IR/DerivedTypes.h>
#include <llvm-16/llvm/IR/Instructions.h>

#include <string>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static LLVMContext TheContext;
static Module *TheModule = new Module("simple function", TheContext);
static IRBuilder<> *B = new IRBuilder<>(TheContext);

void createFunction() {
    // int sum(int*, int n)
    FunctionType *FuncType = FunctionType::get(
        B->getInt32Ty(),
        {PointerType::get(B->getInt32Ty(), 0), B->getInt32Ty()}, false);
    Function *SumFunc =
        Function::Create(FuncType, Function::ExternalLinkage, "sum", TheModule);

    BasicBlock *EntryBB = BasicBlock::Create(TheContext, "entry", SumFunc);
    BasicBlock *LoopBB = BasicBlock::Create(TheContext, "loop", SumFunc);
    BasicBlock *EndBB = BasicBlock::Create(TheContext, "end", SumFunc);

    Value *Nums = SumFunc->getArg(0);
    Value *Total = SumFunc->getArg(1);

    B->SetInsertPoint(EntryBB);
    B->CreateBr(LoopBB);

    B->SetInsertPoint(LoopBB);
    PHINode *SumNode = B->CreatePHI(B->getInt32Ty(), 2);
    PHINode *CountNode = B->CreatePHI(B->getInt32Ty(), 2);

    SumNode->addIncoming(B->getInt32(0), EntryBB);
    CountNode->addIncoming(B->getInt32(0), EntryBB);

    Value *Data = B->CreateLoad(
        B->getInt32Ty(), B->CreateGEP(B->getInt32Ty(), Nums, {CountNode}));
    Value *Sum = B->CreateAdd(SumNode, Data, "sum");

    SumNode->addIncoming(Sum, LoopBB);

    Value *Count = B->CreateAdd(CountNode, B->getInt32(1), "i");
    CountNode->addIncoming(Count, LoopBB);

    Value *Cmp = B->CreateICmpEQ(Count, Total, "cmp");
    B->CreateCondBr(Cmp, EndBB, LoopBB);

    B->SetInsertPoint(EndBB);
    B->CreateRet(Sum);
}

int main(int argc, char *argv[]) {
    createFunction();
    TheModule->print(outs(), nullptr);
    return 0;
}
