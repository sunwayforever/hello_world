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
    FunctionType *FuncType = FunctionType::get(
        B->getVoidTy(), {PointerType::get(B->getInt32Ty(), 0)}, false);
    Function *FooFunc =
        Function::Create(FuncType, Function::ExternalLinkage, "foo", TheModule);

    BasicBlock *Entry = BasicBlock::Create(TheContext, "entry", FooFunc);
    B->SetInsertPoint(Entry);
    Value *S0 =
        B->CreateGEP(B->getInt32Ty(), FooFunc->getArg(0), {B->getInt32(0)});
    Value *S1 =
        B->CreateGEP(B->getInt32Ty(), FooFunc->getArg(0), {B->getInt32(1)});
    B->CreateStore(B->getInt32(1), S0);
    B->CreateStore(B->getInt32(2), S1);
    B->CreateRetVoid();
}

int main(int argc, char *argv[]) {
    createFunction();
    TheModule->print(outs(), nullptr);
    return 0;
}
