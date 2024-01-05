#include <string>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;

static LLVMContext TheContext;
static Module *TheModule = new Module("simple function", TheContext);
static IRBuilder<> *B = new IRBuilder<>(TheContext);

void createBasicFunction() {
    FunctionType *FuncType =
        FunctionType::get(B->getInt32Ty(), {B->getInt32Ty()}, false);
    Function *FooFunc =
        Function::Create(FuncType, Function::ExternalLinkage, "foo", TheModule);

    BasicBlock *Entry = BasicBlock::Create(TheContext, "entry", FooFunc);
    B->SetInsertPoint(Entry);
    // NOTE: global variable
    GlobalVariable *G = dyn_cast_or_null<GlobalVariable>(
        TheModule->getOrInsertGlobal("g", B->getInt32Ty()));
    G->setInitializer(B->getInt32(10));

    Value *GV = B->CreateLoad(B->getInt32Ty(), G);
    // NOTE: alloca
    Value *A = B->CreateAlloca(B->getInt32Ty(), nullptr);
    B->CreateStore(B->getInt32(1), A, false);
    Value *AV = B->CreateLoad(B->getInt32Ty(), A);

    Value *Sum = B->CreateAdd(FooFunc->getArg(0), GV);
    Value *Ret = B->CreateAdd(Sum, AV);
    B->CreateRet(Ret);
}

int main(int argc, char *argv[]) {
    createBasicFunction();
    TheModule->print(outs(), nullptr);
    return 0;
}
