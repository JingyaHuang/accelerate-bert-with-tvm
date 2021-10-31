# TVM passes to realize the optimization
import tvm
from tvm import relay

# Convert PyTorch graph to Relay graph
def pytorch_to_relay(traced_model):  # 0. No pass
	shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in list(traced_model.graph.inputs())[1:]]
	mod, params = relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")  # IR Module
	return mod, params

# 1. FoldConstant
def apply_FoldConstant(mod):
	new_mod = tvm.relay.transform.FoldConstant()(mod)
	return new_mod

# 2. EliminateCommonSubexpr
def apply_EliminateCommonSubexpr(mod):
	new_mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
	return new_mod

# 2bis. Customized EliminateCommonSubexpr (To deal with reshape on static IR)
class ShapeConstDedupMutator(tvm.relay.ExprMutator):
    def __init__(self):
        super().__init__()
        self.shape_consts = {}

    def visit_call(self, call):
        if (isinstance(call.op, tvm.ir.Op) and call.op.name == "reshape"
            and (len(call.args) == 1 or isinstance(call.args[1], tvm.relay.Constant))):
            if len(call.args) > 1:
                assert list(call.attrs.newshape) == list(call.args[1].data.asnumpy())
            new_fn = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            return tvm.relay.Call(new_fn, new_args[:1], call.attrs)
        return super().visit_call(call)

@tvm.relay.transform.function_pass(opt_level=1)
def ShapeConstDedup(fn, mod, ctx):
    return ShapeConstDedupMutator().visit(fn)

def apply_EliminateCommonSubexpr_bis(mod):
	new_mod = ShapeConstDedup(mod)
	new_mod = tvm.relay.transform.EliminateCommonSubexpr()(new_mod)
	return new_mod

# 3. RemoveUnusedFunctions
def apply_RemoveUnusedFunctions(mod):
	new_mod = tvm.relay.transform.RemoveUnusedFunctions()(mod)
	return new_mod

# 4. ToBasicBlockNormalForm
def apply_ToBasicBlockNormalForm(mod):
	new_mod = tvm.relay.transform.ToBasicBlockNormalForm()(mod)
	return new_mod

# 5. FuseOps
def apply_FuseOps(mod):  # Out of memory
	new_mod = tvm.relay.transform.FuseOps(fuse_opt_level=2)(mod) 
	return new_mod

# 6. CombineParallelBatchMatmul
def apply_CombineParallelBatchMatmul(mod):
	new_mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod) 
	return new_mod

# 7. Sequential
def apply_Sequential(mod):
	seq = tvm.transform.Sequential(
	    [
	        relay.transform.FoldConstant(),
	        relay.transform.RemoveUnusedFunctions(),
	        ShapeConstDedup(mod),
	        relay.transform.EliminateCommonSubexpr(),
	        relay.transform.FuseOps(fuse_opt_level=2),
	        relay.transform.ToBasicBlockNormalForm(),
	        tvm.relay.transform.AlterOpLayout(),
	        tvm.relay.transform.CombineParallelBatchMatmul(),
	    ]
	)
	with tvm.transform.PassContext(opt_level=3, disabled_pass=["FuseOps", "AlterOpLayout"]):
		new_mod = seq(mod)

	return new_mod