import inspect
import sys
import os
import numpy
import time
import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')

# ----DL framework----
import torch
import torch.utils.dlpack
import transformers
from transformers import BertModel, BertTokenizer, BertConfig

# ----DL compiler----
import tvm
from tvm import relay, auto_scheduler
import tvm.testing
from tvm import te
from tvm.contrib import graph_runtime, graph_executor

# ----Environment----
# torch.cuda.get_device_name()
# transformers.__version__

# Load pretrained model and its tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

# Freeze weights
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# Tokenizing input text
text = "[CLS] Do you want german potato roll ? [SEP] I prefer not please [SEP]"
# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Creating the trace
traced_model = torch.jit.trace(model, dummy_input).eval()  # No control flow on py, trace to get TorchScript model
for p in traced_model.parameters():
    p.requires_grad_(False)

res_pt = model(tokens_tensor, segments_tensors)

# Check the normal speed
# def y():
#     for i in range(100):
#         model(tokens_tensor, segments_tensors)

# torch_t0 = time.clock()
# y()
# torch_t1 = time.clock()
# torch_time = torch_t1 - torch_t0


# Convert PyTorch graph to Relay graph
shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
mod_bert, params_bert = relay.frontend.pytorch.from_pytorch(traced_model,
                        shape_list, default_dtype="float32")  # IR Module

# Relay Build (compile the graph to llvm target with given input specification)
target = "llvm"  # cuda/opencl/rocm if use GPU
target_host = "llvm"
ctx = tvm.cpu(0)
tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx)
st_a = tvm.nd.array(segments_tensors.numpy(), ctx)

# with tvm.transform.PassContext(opt_level=3):
#     # Output: a relay function
#     graph, lib, params = relay.build(mod_bert, target=target, target_host=target_host, params=params_bert)


# Execute the portable graph on TVM
# module = graph_runtime.create(graph, lib, ctx)
# module.set_input("input_ids", tt_a)
# module.set_input("attention_mask", st_a)
# module.set_input(**params)
# module.run()
# o0 = module.get_output(0)
# o1 = module.get_output(1)
# print(
# 	(numpy.abs((res_pt[0].cpu().numpy() - o0.asnumpy())).max(), 
# 	numpy.abs((res_pt[1].cpu().numpy() - o1.asnumpy())).max())
# 	)


# def x():
#     for i in range(100):
#         module.run()
#     ctx.sync()

# tvm_t0 = time.clock()
# x()
# tvm_t1 = time.clock()
# tvm_time = tvm_t1 - tvm_t0

# Auto-tune the schedulers
# tasks = tvm.autotvm.task.extract_from_program(mod_bert["main"], target=target, params=params)
log_filename = 'bert-tuning.stage1.log'

n_trial = 10  # for real tuning, make this 2000!

def do_tune(tasks, log_filename):
    tmp_log_file = log_filename + ".tmp"
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        tuner = tvm.autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type='curve')
        if os.path.isfile(tmp_log_file):
            tuner.load_history(tvm.autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner.tune(
            n_trial=n_trial,
            early_stopping=600,
            measure_option=tvm.autotvm.measure_option(
                builder=tvm.autotvm.LocalBuilder(timeout=10),
                runner=tvm.autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)),
            callbacks=[
                tvm.autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                tvm.autotvm.callback.log_to_file(tmp_log_file)
            ])

    # pick best records to a cache file
    tvm.autotvm.record.pick_best(tmp_log_file, log_filename)

# do_tune(tasks, log_filename)
# relay.backend.te_compiler.get().clear()

# # Run tuned-tvm again
# with tvm.autotvm.apply_history_best(log_filename):
#     with tvm.transform.PassContext(opt_level=3):
#         graph, lib, params = relay.build(mod_bert,
#                                      target=target,
#                                      target_host=target_host,
#                                      params=params_bert)
# module = graph_runtime.create(graph, lib, ctx)

# module.set_input("input_ids", tt_a)
# module.set_input("attention_mask", st_a)
# module.set_input(**params)
# module.run()
# o0 = module.get_output(0)
# o1 = module.get_output(1)
# print(
# 	(numpy.abs((res_pt[0].cpu().numpy() - o0.asnumpy())).max(), 
# 	numpy.abs((res_pt[1].cpu().numpy() - o1.asnumpy())).max())
# 	)

# def x():
#     for i in range(100):
#         module.run()
#     ctx.sync()

# tvm_tuned_t0 = time.clock()
# x()
# tvm_tuned_t1 = time.clock()
# tvm_tuned_time = tvm_tuned_t1 - tvm_tuned_t0

# Generate DAG graph(origin)


# Now ! TVM optimization passes !
# ---- 0. FoldConstant / EliminateCommonSubexpr
new_mod = tvm.relay.transform.EliminateCommonSubexpr()(mod_bert)
# new_mod = tvm.relay.transform.FoldConstant()(mod_bert)
# new_mod = tvm.relay.transform.FuseOps(fuse_opt_level=2)(mod_bert)  # Out of memory
# new_mod = tvm.relay.transform.RemoveUnusedFunctions()(mod_bert)
# new_mod = tvm.relay.transform.ToBasicBlockNormalForm()(new_mod)
# print(new_mod)

# seq = tvm.transform.Sequential(
#     [
#         relay.transform.FoldConstant(),
#         relay.transform.EliminateCommonSubexpr(),
#         # relay.transform.FuseOps(fuse_opt_level=2),
#     ]
# )
# fused_mod = seq(mod_bert)
# print(fused_mod)
# with tvm.transform.PassContext(opt_level=3):  # disabled_pass=["EliminateCommonSubexpr"]
#     fused_mod = seq(mod_bert)
# print(fused_mod)

# ---- 0.1. Adapt to the hardware
# seq1 = tvm.transform.Sequential([relay.transform.AlterOpLayout()])

# with tvm.transform.PassContext(opt_level=3):
#     with tvm.target.Target("llvm"):
#         hard_mod = seq1(mod_bert)
# print(hard_mod)
# ---- 1. RemoveUnusedFunctions
# ---- 2. ToBasicBlockNormalForm
# ---- 3. EliminateCommonSubexpr

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

# According to Pass Infra, VisitExpr will generate DAG based on Relay tree -> Then DominatorTree
def apply_Sequential(mod):
    seq = tvm.transform.Sequential(
        [
            relay.transform.FoldConstant(),
            relay.transform.RemoveUnusedFunctions(),
            ShapeConstDedup(),
            relay.transform.EliminateCommonSubexpr(),
            relay.transform.FuseOps(fuse_opt_level=2),
            relay.transform.ToBasicBlockNormalForm(),
            tvm.relay.transform.AlterOpLayout()
        ]
    )
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["FuseOps"]):
        new_mod = seq(mod)

    return new_mod

new_mod = apply_Sequential(mod_bert)
print(new_mod)
# print('Relay time(without tune): ', tvm_time / 10.0, 'seconds')
# print('Torch time: ', torch_time / 10.0, 'seconds')
# print('Relay time(tuned): ', tvm_tuned_time / 10.0, 'seconds')


