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
#include <gimple.h>
#include <gimple-iterator.h>
#include <gimple-pretty-print.h>
#include <context.h>
#include <ssa.h>
// clang-format on

#define PASS_NAME test_pass
#define PASS_GIMPLE

void mark_stmt_visited(gimple* stmt) {
    if (gimple_visited_p(stmt)) {
        return;
    }
    gimple_set_visited(stmt, true);
    printf("visit:\n");
    debug(stmt);
    use_operand_p use_p;
    ssa_op_iter i;
    FOR_EACH_PHI_OR_STMT_USE(use_p, stmt, i, SSA_OP_USE) {
        tree use = USE_FROM_PTR(use_p);
        gimple* def = SSA_NAME_DEF_STMT(use);
        mark_stmt_visited(def);
    }
}

unsigned int test_pass_execute() {
    printf("===FNDECL===\n");
    printf("%s\n", function_name(cfun));

    tree var;
    int i;
    FOR_EACH_VEC_SAFE_ELT(cfun->local_decls, i, var) {}

    basic_block bb;
    gimple_stmt_iterator gsi;
    gphi_iterator gpi;
    printf("===BODY===\n");
    FOR_EACH_BB_FN(bb, cfun) {
        for (gsi = gsi_start_bb(bb); !gsi_end_p(gsi); gsi_next(&gsi)) {
            gimple* stmt = gsi_stmt(gsi);
            debug(stmt);
            gimple_set_visited(stmt, false);
        }
        for (gpi = gsi_start_phis(bb); !gsi_end_p(gpi); gsi_next(&gpi)) {
            gimple* stmt = gsi_stmt(gpi);
            debug(stmt);
            gimple_set_visited(stmt, false);
        }
    }

    printf("====================================\n");

    FOR_EACH_BB_FN(bb, cfun)
    for (gsi = gsi_start_bb(bb); !gsi_end_p(gsi); gsi_next(&gsi)) {
        gimple* stmt = gsi_stmt(gsi);
        if (gimple_code(stmt) == GIMPLE_ASSIGN) {
            continue;
        }
        mark_stmt_visited(stmt);
    }

    FOR_EACH_BB_FN(bb, cfun)
    for (gsi = gsi_start_bb(bb); !gsi_end_p(gsi);) {
        gimple* stmt = gsi_stmt(gsi);
        if (!gimple_visited_p(stmt)) {
            printf("remove: \n");
            debug(stmt);
            gsi_remove(&gsi, true);
        } else {
            gsi_next(&gsi);
        }
    }
    return 0;
}

#include "../generate_pass.h"

struct register_pass_info my_passinfo {
    .pass = new test_pass(g), .reference_pass_name = "ssa",
    .ref_pass_instance_number = 1, .pos_op = PASS_POS_INSERT_AFTER
};

void register_callbacks(const char* base_name) {
    register_callback(base_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &my_passinfo);
}
