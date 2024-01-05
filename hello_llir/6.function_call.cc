#include <llvm-16/llvm/IR/DerivedTypes.h>

#include <string>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static LLVMContext TheContext;
static Module *TheModule = new Module("simple function", TheContext);
static IRBuilder<> *B = new IRBuilder<>(TheContext);

void createCalleeFunction() {
    Function *BarFunc = TheModule->getFunction("bar");
    BasicBlock *Entry = BasicBlock::Create(TheContext, "entry", BarFunc);
    B->SetInsertPoint(Entry);
    Value *Ret = B->CreateMul(BarFunc->getArg(0), B->getInt32(2));
    B->CreateRet(Ret);
}

void createCallerFunction() {
    FunctionType *FuncType =
        FunctionType::get(B->getInt32Ty(), {B->getInt32Ty()}, false);
    Function *FooFunc =
        Function::Create(FuncType, Function::ExternalLinkage, "foo", TheModule);

    BasicBlock *Entry = BasicBlock::Create(TheContext, "entry", FooFunc);
    B->SetInsertPoint(Entry);

    Value *Bar = B->CreateCall(TheModule->getFunction("bar"), {B->getInt32(1)});
    Value *Ret = B->CreateMul(FooFunc->getArg(0), Bar);
    B->CreateRet(Ret);
}

int main(int argc, char *argv[]) {
    FunctionType *FuncType =
        FunctionType::get(B->getInt32Ty(), {B->getInt32Ty()}, false);
    Function *BarFunc =
        Function::Create(FuncType, Function::ExternalLinkage, "bar", TheModule);

    createCallerFunction();
    createCalleeFunction();
    TheModule->print(outs(), nullptr);
    return 0;
}
