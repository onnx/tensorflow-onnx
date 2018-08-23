from __future__ import division
from __future__ import print_function

import collections
import tf2onnx
from onnx import numpy_helper, optimizer, TensorShapeProto
from tf2onnx import utils, __version__
from tf2onnx.utils import *

class OnnxNode(object):
    def __init__(self, node, graph):
        self._op = node
        self.graph = graph

        # internel input that would be synced to _op.input once update_proto is called. similiar for output.
        self._input = [i for i in node.input]
        self._output = [i for i in node.output]
        self._attr = {}
        self.inserted_nchw = False

        # dict to original attributes
        for a in node.attribute:
            self._attr[a.name] = a

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def inputs(self):
        val = [self.graph.get_node_by_name(n) for n in self._input]
        return val

    @property
    def attr(self):
        return self._attr

    @property
    def name(self):
        return self._op.name

    @property
    def op(self):
        return self._op

    @name.setter
    def name(self, val):
        self._op.name = val

    @property
    def type(self):
        return self._op.op_type

    def get_output_consumers_at_index(self, output_index):
        output_name = self._output[output_index]
        return self.graph._nodes_outputs[output_name]

    def replace_input(self, old_input_name_with_id, new_input_name_with_id):
        if (":" not in old_input_name_with_id) or (":" not in new_input_name_with_id):
            raise ValueError("input name should contains both node name and output id")

        assert isinstance(old_input_name_with_id, str) and isinstance(new_input_name_with_id, str)
        for i, name in enumerate(self.input):
            if name == old_input_name_with_id:
                self.input[i] = new_input_name_with_id

    def __str__(self):
        return str(self._input) + str(self._output)

    def __repr__(self):
        return "<onnx op type='%s' name=%s>" % (self.type, self._op.name)

    def get_attr(self, name, default=None):
        attr = self.attr.get(name, default)
        return attr

    def set_attr(self, name, value):
        new_attr_val = helper.make_attribute(name, value)
        self.attr[name] = new_attr_val
        new_attrs = [new_attr_val]
        for attr in self._op.attribute:
            if attr.name != name:
                new_attrs.append(attr)
            self._op.attribute.remove(attr)
        
        self._op.attribute.extend(new_attrs)

    def update_proto(self):
        nodes = [n for n in self._op.input]
        for node in nodes:
            self._op.input.remove(node)
        self._op.input.extend(self.input)
        nodes = [n for n in self._op.output]
        for node in nodes:
            self._op.output.remove(node)
        self._op.output.extend(self.output)


class OnnxGraph(object):
    def __init__(self, onnx_graph, enable_debug_log = False):
        self._nodes = []
        self._nodes_by_name = {}
        self._nodes_outputs = {}
        self._node_output_num = {}

        self._initializers = {}

        self._model_inputs = {}
        self._model_outputs = {}

        self.doc_string = onnx_graph.doc_string

        self.enable_log = enable_debug_log

        # model_inputs must be put before add initializer, otherwise the later one will add input.
        # ValueInfoProto
        for n in onnx_graph.input: 
            self._model_inputs[n.name] = n

        for n in onnx_graph.output:
            self._model_outputs[n.name] = n

        # TensorProto
        [self.add_initializer(n) for n in onnx_graph.initializer]
        ops = [OnnxNode(node, self) for node in onnx_graph.node]
        self.set_nodes(ops)

    def log(self, message):
        if self.enable_log:
            print(message)

    def set_nodes(self, ops, delete_nodes_not_referenced = True):
        self._update_node_outputs(ops)
        self._nodes = ops
        if delete_nodes_not_referenced:
            self._remove_uselss_nodes()
            # no need to call self._update_node_outputs(ops), because if a node is not referenced by others, 
            # then it should not exist in the output_num dict.
            self._remove_useless_initializer()

    def _update_node_outputs(self, ops):
        self._nodes_by_name = {op.name: op for op in ops}
        self._nodes_outputs = {}
        self._node_output_num = {}
        for n in ops:
            for input_name in n.input:
                if input_name not in self._nodes_outputs:
                    self._nodes_outputs[input_name] = []
                self._nodes_outputs[input_name].append(self.get_node_by_name(n.name))

                input_node_name = input_name.split(":")[0]
                if input_node_name not in self._node_output_num:
                    self._node_output_num[input_node_name] = 0
                self._node_output_num[input_node_name] += 1

    def _remove_uselss_nodes(self):
        nodes_to_remove = []
        for node in self._nodes:
            if node in nodes_to_remove:
                self.log("node " + node.name + " already in remove list")
                continue

            if (node.name not in self._nodes_outputs or len(self._nodes_outputs[node.name]) == 0) and len(node.input) == 0:
                self.log("removing node [" + node.name + "], since it has no inputs or outputs")
                nodes_to_remove.append(node)

        num = len(nodes_to_remove)
        if num:
            self.log("remove " + str(num) + " nodes(s)")
            for node in nodes_to_remove:
                self._nodes.remove(node)


    def _remove_useless_initializer(self):
        initializer_to_remove = []
        for i_name in self._initializers:
            if i_name in initializer_to_remove:
                self.log("initializer " + i_name + " already in remove list")
                continue

            if i_name not in self._nodes_outputs or len(self._nodes_outputs[i_name]) == 0:
                self.log("removing initializer [" + i_name + "], since it has no inputs or outputs")
                initializer_to_remove.append(i_name)

        num = len(initializer_to_remove)
        if num > 0:
            self.log("remove " + str(num) + " initializer(s)")
            for i_name in initializer_to_remove:
                self.remove_initializer(i_name)

    def has_single_output_node(self, name):
        name = name.split(":")[0]
        return self._node_output_num[name] == 1

    def get_consumer_node_cnt(self, name):
        name = name.split(":")[0]
        if name not in self._node_output_num:
            return 0
        else:
            return self._node_output_num[name]

    def update_proto(self):
        """Update the onnx protobuf from out internal Node structure."""
        for node in self._nodes:
            node.update_proto()

    def get_nodes(self):
        """Get node list."""
        return self._nodes

    def get_node_by_name(self, name):
        """Get node by name."""
        ret = self._nodes_by_name.get(name)
        if not ret and name:
            ret = self._nodes_by_name.get(node_name(name))
        if ret:
            return ret

    def get_initializer_tensor(self, name):
        if self.is_initializer(name):
            return self._initializers[name]
        raise ValueError("no initlializer called" + name)
    
    def update_initializer_tensor(self, name, val):
        #print("update_initializer_tensor " + name)
        if self.is_initializer(name):
            new_tensor = numpy_helper.from_array(val, name)
            if new_tensor.dims != self._initializers[name].dims:
                new_dims = []
                for d in new_tensor.dims:
                    new_dim = TensorShapeProto.Dimension()
                    new_dim.dim_value = d
                    new_dims.append(new_dim)

                del self._model_inputs[name].type.tensor_type.shape.dim[:]
                self._model_inputs[name].type.tensor_type.shape.dim.extend(new_dims)

            del self._initializers[name]
            self._initializers[name] = new_tensor
        else:
            raise ValueError("no initlializer called " + name)

    def remove_initializer(self, name):
        #print("removing initlizer " + name)
        if self.get_consumer_node_cnt(name) != 0:
            raise ValueError("Cannot remove: there are still consumers for the initlializer called " + name)

        if self.is_initializer(name):
            del self._initializers[name]
            if name in self._model_inputs:
                del self._model_inputs[name]
                node_name = name.split(":")[0]
                if node_name in self._node_output_num:
                    del self._node_output_num[name.split(":")[0]]
                if name in self._nodes_outputs:
                    del self._nodes_outputs[name]
                #self._update_node_output_num(self._nodes)
        else:
            raise ValueError("no initlializer called " + name)

    def add_initializer(self, tensor):
        """Add tensor to initializers."""
        #print("add_initializer " + tensor.name)
        self._initializers[tensor.name] = tensor
        if tensor.name not in self._model_inputs:
            self.log("add new input along with adding initlizer " + tensor.name)
            self.log(tensor.dims)
            self._model_inputs[tensor.name] = helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)
            # we don't need update node output num, because both input and initializer created here are not used by other node.
            #self._update_node_output_num(self._nodes)
    
    def is_initializer(self, name):
        return name in self._initializers

    def topological_sort(self, ops):
        """Topological sort of graph."""

        def _push_stack(stack, node, in_stack):
            stack.append(node)
            if node in in_stack:
                raise ValueError('Graph has cycles.')
            else:
                in_stack[node] = True

        def _get_unvisited_child(g, node, not_visited):
            for child in g[node]:
                if child in not_visited:
                    return child
            return -1

        n = len(ops)
        g = [[] for _ in range(n)]
        op_name_to_index = {}
        for i, op in enumerate(ops):
            op_name_to_index[op.name] = i

        for i, op in enumerate(ops):
            for inp in op.input:
                j = self.get_node_by_name(inp)
                if j:
                    g[op_name_to_index[j.name]].append(i)

        # label for each op. highest = sink nodes.
        label = [-1 for _ in range(n)]
        stack = []
        in_stack = dict()
        not_visited = dict.fromkeys([i for i in range(n)])
        label_counter = n - 1

        while not_visited:
            node = list(not_visited.keys())[0]
            _push_stack(stack, node, in_stack)
            while len(stack) > 0:
                node = _get_unvisited_child(g, stack[-1], not_visited)
                if node != -1:
                    _push_stack(stack, node, in_stack)
                else:
                    node = stack.pop()
                    in_stack.pop(node)
                    not_visited.pop(node)
                    label[node] = label_counter
                    label_counter -= 1

        ret = [x for _, x in sorted(zip(label, ops))]
        self.set_nodes(ret)

    def calculate_node(self, description):
        op_cnt = collections.Counter()
        for n in self.get_nodes():
            op_cnt[n.type] += 1
        print(description + ": ops statistics: {}".format(op_cnt))

    def make_model(self, optimize=True):
        inputs = [value for key, value in self._model_inputs.items()]
        initializers = [value for key, value in self._initializers.items()]
        outputs = [value for key, value in self._model_outputs.items()]
        # update attributes
        ops = [n.op for n in self.get_nodes()]

        # create model proto
        graph = helper.make_graph(ops, "tf2onnx",
                                  inputs,
                                  outputs,
                                  initializer=initializers,
                                  doc_string=self.doc_string)

        kwargs = {"producer_name": "tf2onnx",
                  "producer_version": __version__}

        model_proto = helper.make_model(graph, **kwargs)

        # optimize the model proto
        if optimize:
            model_proto = optimizer.optimize(model_proto)
        return model_proto

    @staticmethod
    def replace_subgraph_output(ops, old_output_node, new_output_node):
        # point all children nodes inputs to the new node
        oo = old_output_node
        no = new_output_node
        # we assume the output num of node should be same
        if len(oo.output) != len(no.output):
            raise ValueError("the new node must has same output_num as old node")

        cnt = 0
        for output_name in oo.output:
            index = output_name.split(":")[1]
            for child in ops:
                for i, name in enumerate(child.input):
                    if name == output_name:
                        cnt += 1
                        child.input[i] = no.name + ":" + index
        #print(str(cnt) + " replacement happens")
        return cnt
