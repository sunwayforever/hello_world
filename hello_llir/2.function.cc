#include <string>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static LLVMContext TheContext;
static Module *TheModule = new Module("simple function", TheContext);
static IRBuilder<> *TheBuilder = new IRBuilder<>(TheContext);

void createBasicFunction() {
    FunctionType *FuncType =
        FunctionType::get(TheBuilder->getVoidTy(), {}, false);
    Function *FooFunc =
        Function::Create(FuncType, Function::ExternalLinkage, "foo", TheModule);

    BasicBlock *Entry = BasicBlock::Create(TheContext, "entry", FooFunc);
    TheBuilder->SetInsertPoint(Entry);
    TheBuilder->CreateRetVoid();
}

void createBasicFunctionWithArgs() {
    FunctionType *FuncType = FunctionType::get(
        TheBuilder->getInt32Ty(),
        {TheBuilder->getInt32Ty(), TheBuilder->getInt32Ty()}, false);
    Function *FooFunc = Function::Create(
        FuncType, Function::InternalLinkage, "foo2", TheModule);

    int index = 0;
    Function::arg_iterator AI;
    for (AI = FooFunc->arg_begin(); AI != FooFunc->arg_end(); AI++) {
        AI->setName("arg_" + std::to_string(index++));
    }
    BasicBlock *Entry = BasicBlock::Create(TheContext, "entry", FooFunc);
    TheBuilder->SetInsertPoint(Entry);
    Value *Sum = TheBuilder->CreateAdd(FooFunc->getArg(0), FooFunc->getArg(1));
    Value *Result = TheBuilder->CreateAdd(Sum, TheBuilder->getInt32(1));
    TheBuilder->CreateRet(Result);
}

int main(int argc, char *argv[]) {
    createBasicFunction();
    createBasicFunctionWithArgs();
    TheModule->print(outs(), nullptr);
    return 0;
}
