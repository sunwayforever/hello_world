#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-09-26 12:19
import numpy as np
import tvm
from tvm import relay
from tvm.topi.utils import get_const_tuple


@relay.transform.function_pass(opt_level=1)
class Broadcast:
    def __init__(self, *args):
        self.supported_ops = args

    def transform_function(self, func, mod, ctx):
        obj = self

        class BroadcastTo(tvm.relay.ExprMutator):
            def infer_type(self, node):
                mod = tvm.IRModule.from_expr(node)
                mod = relay.transform.InferType()(mod)
                entry = mod["main"]
                return entry if isinstance(node, relay.Function) else entry.body

            def visit_call(self, call):
                if call.op.name not in obj.supported_ops:
                    return super().visit_call(call)

                if len(call.args) != 2:
                    raise TypeError(
                        f"only 2 args is supported, {call.op.name} have {len(call.args)} args"
                    )
                lhs = self.visit(call.args[0])
                rhs = self.visit(call.args[1])
                lhs_shape = get_const_tuple(self.infer_type(lhs).checked_type.shape)
                rhs_shape = get_const_tuple(self.infer_type(rhs).checked_type.shape)

                dtype = self.infer_type(lhs).checked_type.dtype
                out_shape = np.broadcast(np.empty(lhs_shape), np.empty(rhs_shape)).shape

                if out_shape != lhs_shape:
                    lhs = relay.op.broadcast_to(lhs, out_shape)
                if out_shape != rhs_shape:
                    rhs = relay.op.broadcast_to(rhs, out_shape)

                return relay.expr.Call(
                    call.op, (lhs, rhs), call.attrs, call.type_args, call.span
                )

        return BroadcastTo().visit(func)
