# Visualization of Graph
# Reference: https://tvm.apache.org/2020/07/14/bert-pytorch-tvm
import inspect
import sys
import os
import numpy

# ----DL framework----
import torch
import torch.utils.dlpack

# ----DL compiler----
import tvm
from tvm import relay, te
import graphviz

def compare_latency(avg_time, labels):
    return chart

def visualize(expr, collapse_small=True, node_attr_dict = {}):
    def collect_ops(node):
        ops = set()
        def visitor(e):
            if isinstance(e, tvm.ir.Op):
                ops.add(e.name)
        tvm.relay.analysis.post_order_visit(node, visitor)
        return ops

    # node_dict maps a Relay node to an index (node ID)
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)

    node_dict = {}
    tvm.relay.analysis.post_order_visit(expr, lambda x: _traverse_expr(x, node_dict))

    relayviz_nodes = []

    dot = graphviz.Digraph(format='svg', )
    dot.attr('node', shape = 'box')

    def to_str(node):
        if isinstance(node, tvm.relay.Constant):
            return repr(node).lstrip('Constant(')[:-1]
        else:
            raise NotImplementedError("to_str:" + repr(node))

    def is_small_const(c):
        if not (collapse_small and isinstance(c, tvm.relay.Constant)):
            return False
        if isinstance(c.data, tvm.runtime.ndarray.NDArray):
            return numpy.prod(c.data.shape) < 10
        return True

    # Sort by node ID
    for node, node_id in sorted(node_dict.items(), key=lambda x: x[1]):
        if isinstance(node, tvm.relay.Function):
            dot.node(str(node_id), 'Function', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.body]), str(node_id))
        elif isinstance(node, tvm.relay.Var):
            if node.type_annotation is not None:
                if hasattr(node.type_annotation, 'shape'):
                    shape = tuple([int(x) for x in node.type_annotation.shape])
                    dtype = node.type_annotation.dtype
                    typstr = 'Tensor[{}, {}]'.format(shape, dtype)
                else:
                    typstr = str(node.type_annotation)
            else:
                typstr = '?'
            d = dict(shape = 'ellipse')
            d.update(node_attr_dict.get(node, {}))
            dot.node(str(node_id),
                     '{}: {}'.format(
                         node.name_hint, typstr
                     ), **d)
        elif isinstance(node, tvm.relay.Tuple):
            dot.node(str(node_id), 'Tuple[...])', **node_attr_dict.get(node, {}))
            for field in node.fields:
                dot.edge(str(node_dict[field]), str(node_id))
        elif isinstance(node, tvm.relay.Constant):

            if not is_small_const(node): # small consts are shown in ops
                dot.node(str(node_id), 'Constant({}, {})'.format(node.data.shape, node.data.dtype),
                        **node_attr_dict.get(node, {}))
        elif isinstance(node, tvm.relay.Call):
            args_with_edge = []
            arg_str_list = []
            for arg in node.args:
                if is_small_const(arg):
                    arg_str_list.append(to_str(arg))
                else:
                    arg_str_list.append('·')
                    args_with_edge.append(arg)
            arg_str = ', '.join(arg_str_list)
            if isinstance(node.op, tvm.ir.Op):
                name = node.op.name
                attrs = {k:getattr(node.attrs, k) for k in node.attrs.keys()} if hasattr(node.attrs, 'keys') else {}
                #attrs = inspect.getmembers(node.attrs)
                attr_str_list = [k+'='+(str(v) if len(str(v))<20 else "...") for k, v in attrs.items()]
                if attr_str_list:
                    attr_str = '| '+ ', '.join(attr_str_list)
                else:
                    attr_str = ''
            else:
                ops = collect_ops(node)
                if ops:
                    name = '_'.join(ops)
                else:
                    name = '...'
                attr_str = ''
            s = f'{name}({arg_str}{attr_str})'
            dot.node(str(node_id), s, **node_attr_dict.get(node, {}))
            for arg in args_with_edge:
                dot.edge(str(node_dict[arg]), str(node_id))
        elif isinstance(node, tvm.ir.Op):
            # dot.node(str(node_id), 'Op {}'.format(node.name))
            pass # covered in call
        elif isinstance(node, tvm.relay.TupleGetItem):
            dot.node(str(node_id), 'TupleGetItem(idx={})'.format(node.index), **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.tuple_value]), str(node_id))
        elif isinstance(node, tvm.relay.Let):
            dot.node(str(node_id), 'Let(XX)', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.value]), str(node_id))
            dot.edge(str(node_id), str(node_dict[node.var]))
        else:
            raise RuntimeError(
                'Unknown node type. node_id: {}, node: {}'.format(node_id, type(node)))

    return dot

if __name__ == '__main__':

	# Test on original BERT
	import torch
	import torch.utils.dlpack
	import transformers
	from transformers import BertModel, BertTokenizer, BertConfig

	# Load pretrained model and its tokenizer
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

	# Freeze
	model.eval()
	for p in model.parameters():
	    p.requires_grad_(False)

	# Tokenizing input text
	text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
	tokenized_text = tokenizer.tokenize(text)

	# Masking one of the input tokens
	masked_index = 8
	tokenized_text[masked_index] = '[MASK]'
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

	# Creating a dummy input
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])
	dummy_input = [tokens_tensor, segments_tensors]

	# Creating the trace
	traced_model = torch.jit.trace(model, dummy_input).eval()  # No control flow on py, trace to get TorchScript model
	for p in traced_model.parameters():
	    p.requires_grad_(False)

	res_pt = model(tokens_tensor, segments_tensors)
	
	shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
	mod_bert, params_bert = relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")  # IR Module

	bertGraph = visualize(mod_bert['main'])
	bertGraph.render(filename='bert-origin')
