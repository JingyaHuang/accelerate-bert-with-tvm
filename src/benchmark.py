# BERT TVM benchmark
import os
import time
import argparse
import numpy as np
import inspect
import sys
import os
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
from tvm import relay
from tvm.contrib import graph_runtime

# ----Other scripts----
from utils import visualize
from all_pass import *
from scheduler import auto_tvm_tune, auto_scheduler_tune


parser = argparse.ArgumentParser(description="BERT optimization based on graph compiler")
parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "pass"],
                    help="lauch benchmark or test sigle pass (default: benchmark)")
parser.add_argument("--tvmpass", 
					choices=["original_bert", "autoTVM", "ansor"],
                    default="original_bert",
                    help="choose pass or scheduler for optimization (default: original_bert)")
parser.add_argument("--scheduler", 
					choices=[None, "autoTVM", "ansor"],
                    default=None,
                    help="choose auto-scheduler for optimization (default: None)")
parser.add_argument("--target", type=str, default="llvm", help="default: llvm for cpu")
parser.add_argument("--target_host", type=str, default="llvm", help="default: llvm for cpu")
parser.add_argument("--n_trial", type=int, default=30, help="number of trial for tuning (default: 30)")
parser.add_argument("--repeat", type=int, default=100, help="repeat times of inference (default: 30)")

args = parser.parse_args()

if args.mode=="benchmark":
	print("##########Benchmark BERT starts##########")
elif args.mode=="pass":
	print("##########Benchmark BERT with pass: %s ##########" % args.tvmpass)

######################################################################################
# Load pretrained model and its tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

model.eval()  # Freeze weights
for p in model.parameters():
    p.requires_grad_(False)

######################################################################################
# Prepare test dummy data
# Tokenizing input text
text = "[CLS] Do you want german potato roll ? [SEP] I prefer not please [SEP]"
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

output = model(tokens_tensor, segments_tensors)

######################################################################################
# Graph optimization
mode = args.mode
if mode=="benchmark":
	# Benchmark(General)
	mod, params = pytorch_to_relay(traced_model)
	# FoldConstant
	mod_fc = apply_FoldConstant(mod)
	# EliminateCommonSubexpr
	mod_cse = apply_EliminateCommonSubexpr(mod)
	# Customized EliminateCommonSubexpr (To deal with reshape on static IR)
	mod_cse_bis = apply_EliminateCommonSubexpr_bis(mod)
	# RemoveUnusedFunctions
	mod_ruf = apply_RemoveUnusedFunctions(mod)
	# ToBasicBlockNormalForm
	mod_tbbnf = apply_ToBasicBlockNormalForm(mod)
	# FuseOps
	mod_fo = apply_FuseOps(mod)
	# AlterOpLayout (Adapt to the hardware)
	mod_aol = apply_AlterOpLayout(mod)
	# Sequential
	mod_seq = apply_Sequential(mod)
	mod_list = [mod, mod_fc, mod_cse, mod_cse_bis, mod_ruf, mod_tbbnf, mod_aol, mod_seq]  # Drop mod_fo for FuseOps due to running out of memory

	# Relay Build (compile the graph to llvm target with given input specification)
	target = args.target
	target_host = args.target_host
	ctx = tvm.cpu(0)
	tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx)
	st_a = tvm.nd.array(segments_tensors.numpy(), ctx)

	# lists of graph, lib and params
	graph_list = []
	mean_time = []

	for mod in mod_list:
		with tvm.transform.PassContext(opt_level=3):
			# Output: a relay function
		    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)
		    graph_list.append(graph)

		# Execute the portable graph on TVM
		module = graph_runtime.create(graph, lib, ctx)
		module.set_input("input_ids", tt_a)
		module.set_input("attention_mask", st_a)
		module.set_input(**params)
		# Evaluate inference time cost...
		ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=args.repeat)
		prof_res = np.array(ftimer().results) * 1000 # convert to millisecond
		print("Mean inference time (std dev): %.2f ms (%.2f ms)" %(np.mean(prof_res), np.std(prof_res)))
		mean_time.append(np.mean(prof_res))

	print(mean_time)

	# With schedulers
	# 0.0 AutoTVM without tuning
	# 0.1 Tuned AutoTVM
	# 1.0 Ansor

	# Visualization 
	



# Pass tests

















