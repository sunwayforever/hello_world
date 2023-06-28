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
#include "insn-modes.h"
#include <gimple.h>
#include <gimple-iterator.h>
#include <gimple-pretty-print.h>
#include <context.h>
#include <ssa.h>
#include <rtl.h>
#include <memmodel.h>
#include <emit-rtl.h>
// clang-format on

#define PASS_NAME test_pass
#define PASS_RTL

unsigned int test_pass_execute() {
    const char* fname = function_name(cfun);
    if (strcmp(fname, "trace") == 0) {
        return 0;
    }
    const char* var_name =
        (new std::string("__" + std::string(fname)))->c_str();
    rtx call = gen_rtx_CALL(
        VOIDmode,
        gen_rtx_MEM(FUNCTION_MODE, gen_rtx_SYMBOL_REF(Pmode, "trace")),
        const0_rtx);

    basic_block bb;
    rtx_insn* insn;
    rtx_insn* last_insn;
    for (insn = get_insns(), last_insn = get_last_insn(); insn != last_insn;
         insn = NEXT_INSN(insn)) {
        if (GET_CODE(insn) == NOTE) {
            continue;
        }
        rtx_insn* call_tace = emit_call_insn_before(call, insn);
        rtx_insn* setx = emit_insn_before(
            gen_rtx_SET(
                gen_rtx_REG(DImode, 5), gen_rtx_SYMBOL_REF(DImode, var_name)),
            call_tace);
        goto out;
    }
    /*
    FOR_EACH_BB_FN(bb, cfun) {
        FOR_BB_INSNS(bb, insn) {
            if (GET_CODE(insn) == NOTE) {
                continue;
            }
            rtx_insn* call_tace = emit_call_insn_before(call, insn);
            rtx_insn* setx = emit_insn_before(
                gen_rtx_SET(
                    gen_rtx_REG(DImode, 5),
                    gen_rtx_SYMBOL_REF(DImode, var_name)),
                call_tace);
            goto out;
        }
    }
    */
out:
    return 0;
}
#include "../generate_pass.h"

struct register_pass_info my_passinfo {
    .pass = new test_pass(g), .reference_pass_name = "reload",
    .ref_pass_instance_number = 1, .pos_op = PASS_POS_INSERT_AFTER
};

extern tree pushdecl(tree x);
void callback_parse_function(void* data, void* __unused__) {
    tree t = (tree)data;
    const char* fname = IDENTIFIER_POINTER(DECL_NAME(t));
    tree var_type = build_array_type_nelts(char_type_node, strlen(fname) + 1);
    tree var_decl = build_decl(
        DECL_SOURCE_LOCATION(t), VAR_DECL,
        get_identifier(("__" + std::string(fname)).c_str()), var_type);

    DECL_EXTERNAL(var_decl) = 0;
    TREE_PUBLIC(var_decl) = 1;
    TREE_STATIC(var_decl) = 1;
    DECL_INITIAL(var_decl) = build_string(strlen(fname) + 1, fname);
    TREE_TYPE(DECL_INITIAL(var_decl)) = var_type;
    pushdecl(var_decl);
    varpool_node::finalize_decl(var_decl);
}
void register_callbacks(const char* base_name) {
    register_callback(base_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &my_passinfo);
    register_callback(
        base_name, PLUGIN_START_PARSE_FUNCTION, callback_parse_function, NULL);
}
