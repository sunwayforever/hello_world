#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
using namespace llvm;

static LLVMContext TheContext;

static Module *TheModule = new Module("hello world", TheContext);

int main(int argc, char *argv[]) {
  TheModule->print(outs(), nullptr);
  return 0;
}
