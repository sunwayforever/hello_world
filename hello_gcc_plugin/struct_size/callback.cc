// clang-format off
#include <gcc-plugin.h>
#include <plugin-version.h>
#include <print-tree.h>
#include <tree-core.h>
#include <tree.h>
#include <tree-iterator.h>
#include <tree-pretty-print.h>
#include <iostream>
// clang-format on

// NOTE: 这个 plugin 会打印出 struct 的大小

static void callback_finish_type(void *gcc_data, void *user_data) {
    tree t = (tree)gcc_data;
    if (TREE_CODE(t) != RECORD_TYPE) {
        return;
    }
    // debug_tree(t);
    tree name = TYPE_NAME(t);
    tree size = TYPE_SIZE(t);
    printf("========\n");
    printf(
        "struct %s size: %ld\n", IDENTIFIER_POINTER(name),
        TREE_INT_CST_LOW(size));
}

void register_callbacks(const char *base_name) {
    register_callback(
        base_name, PLUGIN_FINISH_TYPE, callback_finish_type, NULL);
}
