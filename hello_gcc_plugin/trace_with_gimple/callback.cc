// clang-format off
#include <gcc-plugin.h>
#include <plugin-version.h>
#include <print-tree.h>
#include <tree-core.h>
#include <tree.h>
#include <tree-iterator.h>
#include <tree-pretty-print.h>
#include <stringpool.h>
#include <cgraph.h>
#include <gimplify.h>
#include <tree-pass.h>
#include <iostream>
#include "basic-block.h"
#include "dumpfile.h"
#include "plugin.h"
#include <gimple.h>
#include <gimple-iterator.h>
#include <gimple-pretty-print.h>
#include <context.h>
#include <tree-cfg.h>
// clang-format on
tree sancov_fndecl;

static void callback_start_unit(void* gcc_data, void* user_data) {
    tree leaf_attr, nothrow_attr;
    tree BT_FN_VOID = build_function_type_list(
        void_type_node, build_pointer_type(char_type_node), NULL_TREE);
    sancov_fndecl = build_fn_decl("trace", BT_FN_VOID);
    debug_tree(sancov_fndecl);
    // DECL_ASSEMBLER_NAME(sancov_fndecl);
    // TREE_PUBLIC(sancov_fndecl) = 1;
    // DECL_EXTERNAL(sancov_fndecl) = 1;
    // DECL_ARTIFICIAL(sancov_fndecl) = 1;
    // DECL_PRESERVE_P(sancov_fndecl) = 1;
    // DECL_UNINLINABLE(sancov_fndecl) = 1;
    // TREE_USED(sancov_fndecl) = 1;
}

#define PASS_NAME test_pass
#define PASS_GIMPLE

unsigned int test_pass_execute() {
    const char* fname = function_name(cfun);
    if (strcmp(fname, "trace") == 0) {
        return 0;
    }
    basic_block bb;
    FOR_EACH_BB_FN(bb, cfun) {
        gcall* gcall;
        gimple_stmt_iterator gsi = gsi_after_labels(bb);
        if (gsi_end_p(gsi)) continue;
        gimple* stmt = gsi_stmt(gsi);

        gcall = gimple_build_call(
            sancov_fndecl, 1, build_string_literal(strlen(fname) + 1, fname));
        gimple_set_location(gcall, gimple_location(stmt));
        gsi_insert_before(&gsi, gcall, GSI_SAME_STMT);
        break;
    }
    gimple_dump_cfg(stderr, TDF_NONE);
    return 0;
}
#include "../generate_pass.h"

struct register_pass_info my_passinfo {
    .pass = new test_pass(g), .reference_pass_name = "cfg",
    .ref_pass_instance_number = 1, .pos_op = PASS_POS_INSERT_AFTER,
};
void register_callbacks(const char* base_name) {
    register_callback(base_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &my_passinfo);
    register_callback(base_name, PLUGIN_START_UNIT, callback_start_unit, NULL);
}
