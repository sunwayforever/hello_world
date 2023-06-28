// 2023-04-08 10:02
#ifndef GENERATE_PASS_H
#define GENERATE_PASS_H
#include <tree-pass.h>

#ifdef PASS_GIMPLE
#define _PASS_TYPE GIMPLE_PASS
#define _PASS_CLASS gimple_opt_pass
#endif
#ifdef PASS_RTL
#define _PASS_TYPE RTL_PASS
#define _PASS_CLASS rtl_opt_pass
#endif

#define CONCAT2(x, y) x##y
#define STRINGIFY(n) #n

#define __PASS_NAME_NAME(n) STRINGIFY(n)
#define _PASS_NAME_NAME __PASS_NAME_NAME(PASS_NAME)

#define __PASS_NAME_DATA(n) CONCAT2(n, _data)
#define _PASS_NAME_DATA __PASS_NAME_DATA(PASS_NAME)

#define __PASS_EXECUTE(n) CONCAT2(n, _execute)()
#define _PASS_EXECUTE __PASS_EXECUTE(PASS_NAME)

const pass_data _PASS_NAME_DATA = {
    .type = _PASS_TYPE,
    .name = _PASS_NAME_NAME,
    .tv_id = TV_NONE,
    .properties_required = 0,
    .properties_provided = 0,
    .properties_destroyed = 0,
    .todo_flags_start = 0,
    .todo_flags_finish = 0,
};

class PASS_NAME : public _PASS_CLASS {
   public:
    PASS_NAME(gcc::context *ctxt) : _PASS_CLASS(_PASS_NAME_DATA, ctxt) {}
    virtual opt_pass *clone() { return this; }
    virtual unsigned int execute(function *) { return _PASS_EXECUTE; }
};

#endif  // GENERATE_PASS_H
