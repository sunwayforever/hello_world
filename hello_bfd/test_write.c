// 2022-03-04 13:31
#include <assert.h>
#include <bfd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void write_reloc() {
    // int a = 10;
    // int foo() { return a; }
    bfd* abfd = bfd_openw("test_write_reloc.o", "elf64-x86-64");
    bfd_set_arch_mach(abfd, bfd_arch_i386, bfd_mach_x86_64);
    assert(abfd != NULL);
    bfd_set_format(abfd, bfd_object);

    asection* data_section = bfd_make_section_old_way(abfd, ".data");
    asection* text_section = bfd_make_section_old_way(abfd, ".text");

    /* two symbol: foo & a */
    asymbol* symbol_table[3];
    memset(symbol_table, 0, sizeof(symbol_table));

    symbol_table[0] = bfd_make_empty_symbol(abfd);
    symbol_table[0]->name = "foo";
    symbol_table[0]->section = text_section;
    symbol_table[0]->flags = BSF_GLOBAL | BSF_FUNCTION;
    symbol_table[0]->value = 0;

    symbol_table[1] = bfd_make_empty_symbol(abfd);
    symbol_table[1]->name = "a";
    symbol_table[1]->section = data_section;
    symbol_table[1]->flags = BSF_GLOBAL | BSF_OBJECT;
    symbol_table[1]->value = 0;

    bfd_set_symtab(
        abfd, symbol_table, sizeof(symbol_table) / sizeof(symbol_table[0]) - 1);

    /* text */
    char buffer[] = {0xf3, 0x0f, 0x1e, 0xfa, 0x55, 0x48, 0x89, 0xe5,
                     0x8b, 0x05, 0x00, 0x00, 0x00, 0x00, 0x5d, 0xc3};
    int size = sizeof(buffer) / sizeof(buffer[0]);
    bfd_set_section_flags(
        text_section, SEC_CODE | SEC_HAS_CONTENTS | SEC_RELOC);
    bfd_set_section_size(text_section, size);

    /* data */
    bfd_set_section_flags(data_section, SEC_DATA | SEC_HAS_CONTENTS);
    bfd_set_section_size(data_section, 4);
    int x = 0xa;

    bfd_set_section_contents(abfd, text_section, buffer, 0, size);
    bfd_set_section_contents(abfd, data_section, (char*)&x, 0, 4);

    /* reloc */
    /* one reloc: a */
    arelent* reloc_table[2];
    memset(reloc_table, 0, sizeof(reloc_table));
    reloc_table[0] = bfd_alloc(abfd, sizeof(arelent));
    reloc_table[0]->address = 0xa;
    reloc_table[0]->addend = -4;
    /* R_X86_64_PC32 */
    reloc_table[0]->howto = bfd_reloc_type_lookup(abfd, BFD_RELOC_32_PCREL);
    reloc_table[0]->sym_ptr_ptr = &symbol_table[1];
    bfd_set_reloc(
        abfd, text_section, reloc_table,
        sizeof(reloc_table) / sizeof(reloc_table[0]) - 1);
    bfd_close(abfd);
}

void write_symbol() {
    bfd* abfd = bfd_openw("test_write_symbol.o", "elf64-x86-64");
    bfd_set_arch_mach(abfd, bfd_arch_i386, bfd_mach_x86_64);
    assert(abfd != NULL);
    bfd_set_format(abfd, bfd_object);
    asymbol* symbol_table[2];
    memset(symbol_table, 0, sizeof(symbol_table));

    symbol_table[0] = bfd_make_empty_symbol(abfd);
    symbol_table[0]->name = "hello";
    symbol_table[0]->section = bfd_make_section_old_way(abfd, ".text");
    symbol_table[0]->flags = BSF_GLOBAL;
    symbol_table[0]->value = 0;

    bfd_set_symtab(
        abfd, symbol_table, sizeof(symbol_table) / sizeof(symbol_table[0]) - 1);
    bfd_close(abfd);
}

void write_section() {
    /* int foo() {return 10;} */
    bfd* abfd = bfd_openw("test_write_section.o", "elf64-x86-64");
    bfd_set_arch_mach(abfd, bfd_arch_i386, bfd_mach_x86_64);

    assert(abfd != NULL);
    bfd_set_format(abfd, bfd_object);

    /* write symtab */
    asymbol* symbol_table[2];
    memset(symbol_table, 0, sizeof(symbol_table));

    asection* text_section = bfd_make_section_old_way(abfd, ".text");

    symbol_table[0] = bfd_make_empty_symbol(abfd);
    symbol_table[0]->name = "foo2";
    symbol_table[0]->section = text_section;
    symbol_table[0]->flags = BSF_GLOBAL | BSF_FUNCTION;
    symbol_table[0]->value = 0;
    bfd_set_symtab(
        abfd, symbol_table, sizeof(symbol_table) / sizeof(symbol_table[0]) - 1);

    /* write section */
    char buffer[] = {0xf3, 0x0f, 0x1e, 0xfa, 0x55, 0x48, 0x89, 0xe5,
                     0xb8, 0x0a, 0x00, 0x00, 0x00, 0x5d, 0xc3};
    int size = sizeof(buffer) / sizeof(buffer[0]);
    bfd_set_section_flags(text_section, SEC_CODE | SEC_HAS_CONTENTS);
    bfd_set_section_size(text_section, size);
    bfd_set_section_contents(abfd, text_section, buffer, 0, size);
    bfd_close(abfd);
}

int main(int argc, char* argv[]) {
    write_symbol();
    write_section();
    write_reloc();
    return 0;
}
