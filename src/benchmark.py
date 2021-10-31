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
from tvm import relay, auto_scheduler
from tvm.contrib import graph_runtime

# ----Other scripts----
from utils import visualize
from all_pass import *
from scheduler import auto_tvm_tune, auto_scheduler_tune

# Label list of all possible passes
pass_labels = ["FoldConstant", "EliminateCommonSubexpr", "CustomizedEliminateCommonSubexpr",
 		  		"RemoveUnusedFunctions", "ToBasicBlockNormalForm", "CombineParallelBatchMatmul", 
 		  		"Sequential"]

parser = argparse.ArgumentParser(description="BERT optimization based on graph compiler")
parser.add_argument("--mode", type=str, default="pass", choices=["benchmark", "pass"],
                    help="lauch benchmark or test sigle pass (default: pass)")
parser.add_argument("--tvmpass", 
					choices=[None]+pass_labels,
                    default=None,
                    help="choose pass for optimization (default: None)")
parser.add_argument("--scheduler", 
					choices=[None, "AutoTVM", "ANSOR"],
                    default=None,
                    help="choose scheduler for optimization (default: None)")
parser.add_argument("--target", type=str, default="llvm", help="default: llvm for cpu")
parser.add_argument("--target_host", type=str, default="llvm", help="default: llvm for cpu")
parser.add_argument("--n_trial", type=int, default=30, help="number of trial for tuning (default: 30)")
parser.add_argument("--repeat", type=int, default=100, help="repeat times of inference (default: 30)")

args = parser.parse_args()
	

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
# Relay Build Config
target = args.target
target_host = args.target_host
ctx = tvm.cpu(0)
tt_a = tvm.nd.array(tokens_tensor.numpy(), ctx)
st_a = tvm.nd.array(segments_tensors.numpy(), ctx)

# Execute the portable graph on TVM
def exe_graph(graph, lib, ctx, tt_a, st_a, params, repeat):
	module = graph_runtime.create(graph, lib, ctx)
	module.set_input("input_ids", tt_a)
	module.set_input("attention_mask", st_a)
	module.set_input(**params)
	# Evaluate inference time cost...
	ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=repeat)
	prof_res = np.array(ftimer().results) * 1000 # convert to millisecond

	print("Mean inference time (std dev): %.2f ms (%.2f ms)" %(np.mean(prof_res), np.std(prof_res)))

	return np.mean(prof_res)

# Relay build
if mode=="pass":
	mod, params = pytorch_to_relay(traced_model)
	if args.tvmpass:
		print("##########Benchmark BERT with pass: %s ##########" % args.tvmpass)
		# Apply TVM passes 
		pass_dict = {"FoldConstant": apply_FoldConstant, 
					 "EliminateCommonSubexpr": apply_EliminateCommonSubexpr, 
					 "CustomizedEliminateCommonSubexpr": apply_EliminateCommonSubexpr_bis,
					 "RemoveUnusedFunctions": apply_RemoveUnusedFunctions, 
					 "ToBasicBlockNormalForm": apply_ToBasicBlockNormalForm, 
					 "CombineParallelBatchMatmul": apply_CombineParallelBatchMatmul, 
					 "Sequential": apply_Sequential,
					 }
		mod = pass_dict[args.tvmpass](mod)

		# Clear compile engine
		relay.backend.te_compiler.get().clear()

		with tvm.transform.PassContext(opt_level=3):
			# Output: a relay function
		    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

		# Execute the portable graph on TVM
		exe_graph(graph, lib, ctx, tt_a, st_a, params, args.repeat)
		modGraph = visualize(mod['main'])
		modGraph.render(filename=f'img/bert-{args.tvmpass}')
	else:
		print("########## Launch witout optimization ##########")
		# Use orginal relay from pytorch
		with tvm.transform.PassContext(opt_level=3):
			# Output: a relay function
		    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

		# Execute the portable graph on TVM
		exe_graph(graph, lib, ctx, tt_a, st_a, params, args.repeat)
		modGraph = visualize(mod['main'])
		modGraph.render(filename=f'img/bert-original')
		   
	# Apply scheduler
	if args.scheduler:
		print("##########Benchmark BERT with scheduler: %s ##########" % args.scheduler)
		if args.scheduler=="AutoTVM":
			# Create task
			tasks = tvm.autotvm.task.extract_from_program(mod["main"], target=target, params=params)
			log_filename = 'logs/autotvm-bert-tuning.stage1.log'
			n_trial = args.n_trial
			# Tune AutoTVM
			auto_tvm_tune(tasks, log_filename, n_trial)
			relay.backend.te_compiler.get().clear()

			# Compile with tuned-tvm
			with tvm.autotvm.apply_history_best(log_filename):
			    with tvm.transform.PassContext(opt_level=3):
			        graph, lib, params = relay.build(mod,
			                                     target=target,
			                                     target_host=target_host,
			                                     params=params)
			exe_graph(graph, lib, ctx, tt_a, st_a, params, args.repeat)

		elif args.scheduler=="ANSOR":
			# Tune Ansor

			tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
			log_filename = 'logs/ansor-bert-tuning.stage1.log'
			n_trial = args.n_trial
			# Tune Ansor
			auto_scheduler_tune(tasks, task_weights, log_filename, n_trial)
			relay.backend.te_compiler.get().clear()

			# Compile with tuned-ansor again
			with auto_scheduler.ApplyHistoryBest(log_filename):
			    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
			        graph, lib, params = relay.build(mod,
			                                     target=target,
			                                     target_host=target_host,
			                                     params=params)
			exe_graph(graph, lib, ctx, tt_a, st_a, params, args.repeat)

elif mode=="benchmark":
	print("##########Benchmark BERT starts##########")
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
	mod_cpb = apply_CombineParallelBatchMatmul(mod)
	# Sequential
	mod_seq = apply_Sequential(mod)
	mod_list = [mod, mod_fc, mod_cse, mod_cse_bis, mod_ruf, mod_tbbnf, mod_cpb, mod_seq]  # Drop mod_fo for FuseOps due to running out of memory

	# lists of labels, graph and avg. latency time
	graph_list = []
	mean_time_list = []

	for mod in mod_list:
		with tvm.transform.PassContext(opt_level=3):
			# Output: a relay function
		    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)
		    graph_list.append(graph)

		# Clear compile engine
		relay.backend.te_compiler.get().clear()

		# Execute the portable graph on TVM
		mean_time = exe_graph(graph, lib, ctx, tt_a, st_a, params, args.repeat)
		mean_time_list.append(mean_time)
		modGraph = visualize(mod['main'])
		modGraph.render(filename=f'img/bert-original')

	all_labels = ['PyTorch'] + pass_labels + ["AutoTVM", "ANSOR"]

	# With schedulers
	# 0.0 AutoTVM 
	tasks = tvm.autotvm.task.extract_from_program(mod["main"], target=target, params=params)
	log_filename = 'logs/autotvm-bert-tuning.stage1.log'
	n_trial = args.n_trial
	# Tune AutoTVM
	auto_tvm_tune(tasks, log_filename, n_trial)
	relay.backend.te_compiler.get().clear()

	# Compile with tuned-tvm
	with tvm.autotvm.apply_history_best(log_filename):
	    with tvm.transform.PassContext(opt_level=3):
	        graph, lib, params = relay.build(mod,
	                                     target=target,
	                                     target_host=target_host,
	                                     params=params)
	mean_time = exe_graph(graph, lib, ctx, tt_a, st_a, params, args.repeat)
	mean_time_list.append(mean_time)
	# 1.0 Ansor
	tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
	log_filename = 'logs/ansor-bert-tuning.stage1.log'
	n_trial = args.n_trial
	# Tune Ansor
	auto_scheduler_tune(tasks, task_weights, log_filename, n_trial)
	relay.backend.te_compiler.get().clear()

	# Compile with tuned-ansor again
	with auto_scheduler.ApplyHistoryBest(log_filename):
		with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                graph, lib, params = relay.build(mod, target=target, params=params)

	mean_time = exe_graph(graph, lib, ctx, tt_a, st_a, params, args.repeat)
	mean_time_list.append(mean_time)




