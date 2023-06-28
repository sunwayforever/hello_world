// clang-format off
#include <gcc-plugin.h>
#include <plugin-version.h>
#include <print-tree.h>
#include <tree-core.h>
#include <tree.h>
#include <tree-iterator.h>
#include <tree-pretty-print.h>
#include <iostream>
#include <stringpool.h>
#include "plugin.h"

// clang-format on
static tree handle_notrace_attribute(
    tree *node, tree name, tree args, int flags, bool *no_add_attrs) {
    return NULL_TREE;
}

static struct attribute_spec notrace_attr = {
    "notrace", 0, 0, false, false, false, false, handle_notrace_attribute};

void callback_register_attribute(void *data, void *__unused__) {
    register_attribute(&notrace_attr);
}

void callback_parse_function(void *data, void *__unused__) {
    std::cerr << "======" << std::endl;
    std::cerr << "FUNCTION: " << std::endl;
    tree t = (tree)data;
    tree attributes = DECL_ATTRIBUTES(t);
    if (attributes) {
        for (tree t = attributes; t != NULL_TREE; t = TREE_CHAIN(t)) {
            if (TREE_PURPOSE(t) == get_identifier("notrace")) {
                return;
            }
        }
    }
    const char *fname = IDENTIFIER_POINTER(DECL_NAME(t));
    if (strcmp(fname, "trace") == 0) {
        return;
    }
    printf("%s %s\n", get_tree_code_name(TREE_CODE(t)), fname);

    t = DECL_SAVED_TREE(t);
    tree ftype = build_function_type_list(
        void_type_node, build_pointer_type(char_type_node), NULL_TREE);
    tree fndecl = build_fn_decl("trace", ftype);
    if (TREE_CODE(t) == STATEMENT_LIST) {
        // NOTE: it seems `main` starts wil `statement_list`, while other
        // function starts with `bind_expr`
        tree_stmt_iterator iter = tsi_start(t);
        t = tsi_stmt(iter);
    }
    tree body = BIND_EXPR_BODY(t);
    if (TREE_CODE(body) != STATEMENT_LIST) {
        tree tmp = alloc_stmt_list();
        append_to_statement_list(body, &tmp);
        body = tmp;
        BIND_EXPR_BODY(t) = body;
    }
    tree_stmt_iterator iter = tsi_start(body);
    tree call_stmt = build_call_expr(
        fndecl, 1, build_string_literal(strlen(fname) + 1, fname));
    tsi_link_before(&iter, call_stmt, TSI_SAME_STMT);
}

void register_callbacks(const char *base_name) {
    register_callback(
        base_name, PLUGIN_FINISH_PARSE_FUNCTION, callback_parse_function, NULL);
    register_callback(
        base_name, PLUGIN_ATTRIBUTES, callback_register_attribute, NULL);
}
