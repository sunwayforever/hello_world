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
#include <gimple.h>
#include <gimple-iterator.h>
#include <gimple-pretty-print.h>
#include <context.h>
// clang-format on

static void callback_pre_genericize(void *gcc_data, void *user_data) {
    printf("======= AST ======\n");
    tree t = (tree)gcc_data;
    debug_tree(DECL_SAVED_TREE(t));
    print_generic_decl(stderr, t, TDF_RAW);
    print_generic_stmt(stderr, DECL_SAVED_TREE(t), TDF_RAW);
}

static void callback_finish_parse_function(void *gcc_data, void *user_data) {
    printf("======= GENERIC ======\n");
    tree t = (tree)gcc_data;
    print_generic_decl(stderr, t, TDF_RAW);
    print_generic_stmt(stderr, DECL_SAVED_TREE(t), TDF_RAW);
    debug_tree(DECL_SAVED_TREE(t));
}

const pass_data my_pass_data = {
    .type = GIMPLE_PASS,
    .name = "my_pass",
    .optinfo_flags = OPTGROUP_NONE,
    .tv_id = TV_NONE,
    .properties_required = PROP_gimple_any,
    .properties_provided = 0,
    .properties_destroyed = 0,
    .todo_flags_start = 0,
    .todo_flags_finish = 0,
};

#define PASS_NAME test_pass
#define PASS_GIMPLE

unsigned int test_pass_execute() {
    printf("===GIMPLE===\n");
    basic_block bb;
    gimple_stmt_iterator gsi;

    FOR_EACH_BB_FN(bb, cfun) {
        printf("---BB---\n");
        for (gsi = gsi_start_bb(bb); !gsi_end_p(gsi); gsi_next(&gsi)) {
            gimple *stmt = gsi_stmt(gsi);
            print_gimple_stmt(stderr, stmt, 0, TDF_RAW);
            printf("gimple code: %s\n", gimple_code_name[gimple_code(stmt)]);
            if (gimple_code(stmt) == GIMPLE_RETURN) {
                printf("return:\n");
                debug_tree(gimple_return_retval((greturn *)stmt));
            } else {
                printf("lhs:\n");
                debug_tree(gimple_get_lhs(stmt));
                printf("rhs1:\n");
                debug_tree(gimple_assign_rhs1(stmt));
            }
        }
    }
    return 0;
}
#include "../generate_pass.h"

struct register_pass_info my_passinfo {
    .pass = new test_pass(g), .reference_pass_name = "cfg",
    .ref_pass_instance_number = 1, .pos_op = PASS_POS_INSERT_AFTER,
};

void register_callbacks(const char *base_name) {
    register_callback(
        base_name, PLUGIN_PRE_GENERICIZE, callback_pre_genericize, NULL);
    register_callback(
        base_name, PLUGIN_FINISH_PARSE_FUNCTION, callback_finish_parse_function,
        NULL);
    register_callback(base_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &my_passinfo);
}
