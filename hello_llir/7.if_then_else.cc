#include <llvm-16/llvm/IR/DerivedTypes.h>

#include <string>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static LLVMContext TheContext;
static Module *TheModule = new Module("simple function", TheContext);
static IRBuilder<> *B = new IRBuilder<>(TheContext);

void createFunction() {
    FunctionType *FuncType =
        FunctionType::get(B->getInt32Ty(), {B->getInt32Ty()}, false);
    Function *FooFunc =
        Function::Create(FuncType, Function::ExternalLinkage, "foo", TheModule);
    BasicBlock *Entry = BasicBlock::Create(TheContext, "entry", FooFunc);
    BasicBlock *Then = BasicBlock::Create(TheContext, "then", FooFunc);
    BasicBlock *Else = BasicBlock::Create(TheContext, "else", FooFunc);
    BasicBlock *Merge = BasicBlock::Create(TheContext, "merge", FooFunc);
    B->SetInsertPoint(Entry);
    Value *Tmp = B->CreateAlloca(B->getInt32Ty(), nullptr);
    Value *Cmp = B->CreateICmpEQ(FooFunc->getArg(0), B->getInt32(0));
    B->CreateCondBr(Cmp, Then, Else);
    // then
    B->SetInsertPoint(Then);
    B->CreateStore(B->getInt32(1), Tmp);
    B->CreateBr(Merge);
    // else
    B->SetInsertPoint(Else);
    B->CreateStore(B->getInt32(2), Tmp);
    B->CreateBr(Merge);
    // merge
    B->SetInsertPoint(Merge);
    Value *Ret = B->CreateLoad(B->getInt32Ty(), Tmp);
    B->CreateRet(Ret);
}

int main(int argc, char *argv[]) {
    createFunction();
    TheModule->print(outs(), nullptr);
    return 0;
}
