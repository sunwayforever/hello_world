
#include <string>

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static LLVMContext TheContext;
static Module *TheModule = new Module("simple function", TheContext);
static IRBuilder<> *B = new IRBuilder<>(TheContext);

Function *createSumFunction() {
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
    return SumFunc;
}

int main(int argc, char *argv[]) {
    Function *F = createSumFunction();
    ExecutionEngine *EE =
        EngineBuilder(std::unique_ptr<Module>(TheModule)).create();
    std::vector<GenericValue> Args;
    GenericValue X;
    int32_t data[5] = {1, 2, 3, 4, 5};
    X.PointerVal = reinterpret_cast<void *>(data);
    Args.push_back(X);
    GenericValue Y;
    Y.IntVal = APInt(32, 5);
    Args.push_back(Y);
    GenericValue gv = EE->runFunction(F, Args);
    outs() << "sum: " << gv.IntVal << "\n";
    return 0;
}
