// 2022-03-04 11:42
#include <assert.h>
#include <bfd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void read_reloc(bfd *abfd) {
    printf("---------- %s ----------\n", __FUNCTION__);

    int symtab_size = bfd_get_symtab_upper_bound(abfd);
    assert(symtab_size > 0);
    asymbol **symbol_table = (asymbol **)malloc(symtab_size);
    bfd_canonicalize_symtab(abfd, symbol_table);

    asection *text_section = bfd_get_section_by_name(abfd, ".text");
    int reloc_size = bfd_get_reloc_upper_bound(abfd, text_section);
    arelent **reloc_table = (arelent **)malloc(reloc_size);
    int num_reloc =
        bfd_canonicalize_reloc(abfd, text_section, reloc_table, symbol_table);
    printf("reloc_size: %d, num_reloc: %d\n", reloc_size, num_reloc);
    for (int i = 0; i < num_reloc; i++) {
        arelent *reloc = reloc_table[i];
        printf(
            "offset: 0x%lx rel: %-20s symbol: %-10s addend: %ld\n",
            reloc->address, reloc->howto->name, (*(reloc->sym_ptr_ptr))->name,
            reloc->addend);
    }
}

void read_symtab(bfd *abfd) {
    printf("---------- %s ----------\n", __FUNCTION__);
    int symtab_size = bfd_get_symtab_upper_bound(abfd);
    assert(symtab_size > 0);

    asymbol **symbol_table = (asymbol **)malloc(symtab_size);
    int num_symbols = bfd_canonicalize_symtab(abfd, symbol_table);
    printf("symtabl size: %d, num of symbols: %d\n", symtab_size, num_symbols);

    symbol_info info;
    for (int i = 0; i < num_symbols; i++) {
        if (symbol_table[i]->section == NULL) {
            continue;
        }
        bfd_symbol_info(symbol_table[i], &info);
        printf(
            "section: %-20s, symbol: %-25s -> 0x%lx, type: %x\n",
            symbol_table[i]->section->name, info.name, info.value, info.type);
    }
}

void read_section(bfd *abfd) {
    asection *section = bfd_get_section_by_name(abfd, ".text");
    printf(".text vma: 0x%lx\n", section->vma);

    char *buffer = (char *)malloc(section->size);
    bfd_get_section_contents(abfd, section, buffer, 0, section->size);
    for (int i = 0; i < section->size; i++) {
        printf("%.2x ", buffer[i] & 0xff);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    bfd_init();
    /* bfd_openr 用来读 */
    /* bfd_openw 用来写 */
    bfd *abfd = bfd_openr("test.obj", NULL);
    printf("target: %s\n", bfd_get_target(abfd));

    assert(abfd != NULL);
    /* abfd 支持三种格式, 其中 elf 的 executable, o, so 都算 bfd_object */
    /*
     * if (bfd_check_format(abfd, bfd_archive)) {
     *     printf("found archive\n");
     * }
     * if (bfd_check_format(abfd, bfd_core)) {
     *     printf("found core\n");
     * }
     */
    assert(bfd_check_format(abfd, bfd_object) != 0);

    read_symtab(abfd);
    read_reloc(abfd);
    read_section(abfd);
    return 0;
}
