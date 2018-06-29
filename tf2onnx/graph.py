# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.graph - class to manage graph manipulation on top of onnx
"""
from onnx import numpy_helper, optimizer, ModelProto, defs, OperatorSetIdProto
from tf2onnx import utils, tfonnx, __version__
from tf2onnx.utils import *


class Node(object):
    """A Node - wrapper around onnx nodes that we use for graph manipulations."""

    def __init__(self, node, graph):
        """Create Node.
        Args:
            node: Onnx node in NodeProto
            graph: Graph() we are part of
        """
        self._op = node
        self.graph = graph
        self._input = [i for i in node.input]
        self._output = [i for i in node.output]
        self._attr = {}
        self.inserted_nchw = False
        # make sure this name is not used
        assert graph.get_node_by_name(node.name) is None
        graph.set_node_by_name(self)
        # dict to original attributes
        for a in node.attribute:
            self._attr[a.name] = a
        # try to find a dtype for this node
        dtype = graph.dtypes.get(node.name)
        if not dtype:
            dtype = self._attr.get("dtype")
            if dtype:
                dtype = dtype.i
        self._dtype = dtype
        self.data_format = self.get_attr("data_format")
        if self.data_format:
            self.data_format = self.data_format.s.decode("utf-8")

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def inputs(self):
        """Input node objects."""
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
        """Return Op type."""
        return self._op.op_type

    @type.setter
    def type(self, val):
        """Set Op type."""
        self._op.op_type = val

    @property
    def domain(self):
        """Return Op type."""
        return self._op.domain

    @type.setter
    def domain(self, val):
        """Set Op type."""
        self._op.domain = val

    def is_nhwc(self):
        """Return True if node is in NCHW format."""
        return self.data_format == "NHWC"

    def is_const(self):
        """Return True if node is a constant."""
        return self.type in ["Const", "ConstV2"]

    def __str__(self):
        return str(self._op)

    def __repr__(self):
        return "<onnx op type='%s' name=%s>" % (self.type, self._op.name)

    def get_attr(self, name):
        """Get attribute map."""
        return self.attr.get(name)

    def set_attr(self, name, value):
        self.attr[name] = helper.make_attribute(name, value)

    def set_deleted(self):
        self.type = "@@DELETED@@"

    def is_deleted(self):
        return self.type == "@@DELETED@@"

    @property
    def shape(self):
        """Get shape for now."""
        shape = self.get_attr("shape")
        if shape:
            shape = shape.ints
            # TODO: this is what we want ?
            if shape and shape[0] == -1:
                shape[0] = utils.ONNX_UNKNOWN_DIMENSION
        return shape

    def get_tensor_type(self):
        """Get the onnx data type of a tensor."""
        t = self.get_attr("value")
        if t:
            t = helper.get_attribute_value(t)
            if t:
                return utils.ONNX_TO_NUMPY_DTYPE[t.data_type]
        return onnx_pb.TensorProto.FLOAT

    def get_tensor_value(self):
        """Get value for onnx tensor."""
        if not self.is_const():
            raise ValueError("get tensor value: {} must be Const".format(self.name))

        t = self.get_attr("value")
        if t:
            t = helper.get_attribute_value(t)
            if t.raw_data:
                buf = np.frombuffer(t.raw_data,
                                    dtype=utils.ONNX_TO_NUMPY_DTYPE[t.data_type]).reshape(t.dims)
                return buf
            if t.int32_data:
                return t.int32_data
            if t.int64_data:
                return t.int64_data
            if t.float_data:
                return t.float_data
            raise ValueError("tensor data_type not handled in get_tensor_value")
            
    def get_tensor(self):
        if not self.is_const():
            if self.type == "Identity":
                return self.inputs[0].get_tensor()
            raise ValueError("get tensor: {} must be Const".format(self.name))
        t = self.get_attr("value")
        if t:
            t = numpy_helper.to_array(helper.get_attribute_value(t))
        return t

    def scalar_to_dim1(self):
        """Get value for onnx tensor."""
        if not self.is_const():
            raise ValueError("get tensor value: {} must be Const".format(self.name))

        t = self.get_attr("value")
        if t:
            t = helper.get_attribute_value(t)
            if len(t.dims) == 0:
                t.dims.extend([1])
        return t.dims

    def set_tensor_value(self, new_val):
        """Set new value for existing onnx tensor."""
        if not self.is_const():
            raise ValueError("get tensor value: {} must be Const".format(self.name))
        t = self.get_attr("value")
        if not t:
            raise ValueError("set tensor value: {} is None".format(self.name))
        t = helper.get_attribute_value(t)
        if not t.raw_data:
            raise ValueError("set tensor value: {} is not raw_data".format(self.name))
        t.raw_data = new_val.tobytes()
        for i, v in enumerate(t.dims):
            t.dims[i] = new_val.shape[i]
        # track shapes in _output_shapes
        self.graph.set_shape(t.name, t.dims)

    @property
    def dtype(self):
        """Return dtype."""
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        """Set dtype."""
        self._dtype = val

    def update_proto(self):
        """Update protobuf from internal structure."""
        nodes = [n for n in self._op.input]
        for node in nodes:
            self._op.input.remove(node)
        self._op.input.extend(self.input)
        nodes = [n for n in self._op.output]
        for node in nodes:
            self._op.output.remove(node)
        self._op.output.extend(self.output)


class Graph(object):
    """"Class that provides graph manipulation and matching."""

    def __init__(self, nodes, output_shapes=None, dtypes=None, target=None, opset=None, extra_opset=None):
        """Create Graph.
        Args:
            nodes: list of Node()
            output_shapes: dict of tensorflow output shapes
            dtypes: dict of tensorflow dtype
        """
        if target is None:
            target = tfonnx.DEFAULT_TARGET
        self._nodes = []
        self._initializers = {}
        self._nodes_by_name = {}
        self.shapes = {}
        self.model_inputs = []
        self._target = set(target)
        self.dtypes = dtypes
        self._output_shapes = output_shapes
        ops = [Node(node, self) for node in nodes]
        self.set_nodes(ops)
        if opset is None or opset == 0:
            opset = defs.onnx_opset_version()
        self._opset = opset
        self._extra_opset = extra_opset

    @property
    def opset(self):
        return self._opset

    def is_target(self, name):
        """Return True if target platform is name."""
        return name in self._target

    def make_const(self, name, op_type, val):
        """Make a new constant in the graph."""
        onnx_tensor = numpy_helper.from_array(val, name)
        self._initializers[name] = onnx_tensor
        self.set_shape(name, val.shape)
        new_node = Node(helper.make_node(op_type, [], [name], name=name, value=onnx_tensor), self)
        return new_node

    def set_nodes(self, ops):
        """Set new node list."""
        self._nodes = ops
        self._nodes_by_name = {op.name: op for op in ops}

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
        return ret

    def set_node_by_name(self, node):
        """Set node by name."""
        self._nodes_by_name[node.name] = node

    def add_initializer(self, tensor):
        """Add tensor to initializers."""
        self._initializers[tensor.name] = tensor

    def get_shape(self, name):
        """Get shape for node."""
        assert isinstance(name, str)
        shape = self._output_shapes.get(name)
        if shape:
            for i, v in enumerate(shape):
                if v is None:
                    shape[i] = -1
            if shape[0] == -1:
                shape[0] = utils.ONNX_UNKNOWN_DIMENSION
        return shape

    def set_shape(self, name, val):
        """Set new shape of node."""
        if type(val) is np.ndarray:
            val = val.tolist()
        self._output_shapes[name] = val

    def copy_shape(self, input_name, output_name):
        """Copy shape from another node."""
        shape = self.get_shape(input_name)
        # assert shape is not None
        if shape:
            self.set_shape(output_name, shape)

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

    def make_model(self, doc, input_names, output_names, optimize=True):
        """
        Create final ModelProto for onnx from internal graph.
        Args:
            optimize: optimize graph via onnx
            doc: text for doc string of the model
            input_names: list of model inputs
            output_names: list of model outputs
        """

        # create output_tensor_values
        output_tensor_values = []
        for name in output_names:
            op = self.get_node_by_name(name)
            if op:
                dtype = op.dtype
                if not dtype:
                    continue
                v = helper.make_tensor_value_info(name, dtype, self.get_shape(name))
                output_tensor_values.append(v)

        # update attributes
        ops = []
        all_inputs = set()
        for op in self.get_nodes():
            all_inputs |= set(op.input)
            onnx_op = op.op
            del onnx_op.attribute[:]
            attr = []
            for a in op.attr.values():
                if a.name in utils.ONNX_VALID_ATTRIBUTES:
                    attr.append(a)
            if attr:
                onnx_op.attribute.extend(attr)
            ops.append(onnx_op)

        # create input_tensor_values, initializers
        initializers = [i for i in list(self._initializers.values()) if i.name in all_inputs]
        input_with_initializers = []
        for initializer in initializers:
            shape = self.get_shape(initializer.name)
            if shape and list(shape) != initializer.dims:
                raise ValueError("initializer shape is inconsistent")
            val = helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims)
            input_with_initializers.append(val)
        input_with_initializers.extend(self.model_inputs)

        # create model proto
        graph = helper.make_graph(ops, "tf2onnx",
                                  input_with_initializers,
                                  output_tensor_values,
                                  initializer=initializers,
                                  doc_string=doc)

        kwargs = {"producer_name": "tf2onnx",
                  "producer_version": __version__}
        opsets = []
        imp = OperatorSetIdProto()
        imp.version = self._opset
        opsets.append(imp)
        if self._extra_opset is not None:
            opsets.extend(self._extra_opset)
        kwargs["opset_imports"] = opsets

        model_proto = helper.make_model(graph, **kwargs)

        # optimize the model proto
        if optimize:
            model_proto = optimizer.optimize(model_proto)
        return model_proto

    def dump_graph(self):
        """Dump graph with shapes (helpful for debugging)."""
        for node in self.get_nodes():
            input_names = ["{}{}".format(n, self.get_shape(n)) for n in node.input]
            print("{} {} {} {}".format(node.type, self.get_shape(node.output[0]), node.name, ", ".join(input_names)))

    def follow_inputs(self, node, num, space=""):
        """Follow inputs for (helpful for debugging)."""
        val = []
        top = space == ""
        if num == 0:
            return []
        val.append("{}{} {} {}".format(space, node.type, node.name, self.get_shape(node.name + ":0")))
        space += "    "
        for j in node.inputs:
            val.extend(self.follow_inputs(j, num - 1, space))
        if top:
            print("\n".join(reversed(val)))
            print()
        else:
            return val

    @staticmethod
    def remove_input(node, to_be_removed):
        """Remove input from Node.
        Args:
            node: the node we expect the input on
            to_be_removed: the node name we want to remove
        """
        assert isinstance(node, Node) and isinstance(to_be_removed, str)
        for i, name in enumerate(node.input):
            if name == to_be_removed:
                del node.input[i]
                break
        # don't remove output from parent since others might depend on it
        return True

    def insert_new_node_on_input(self, node, op_type, input_name, name=None, **kwargs):
        """Create and insert a new node into the graph.
        Args:
            node: we want to replace the input for this node
            op_type: type for new operation
            input_name: the names of the outputs above us
            name: the name of the new op
            kwargs: attributes of the new node

        Returns:
            node that was inserted
        """
        if name is None:
            name = utils.make_name(node.name)
        new_output = name + ":0"
        new_node = Node(helper.make_node(op_type, [input_name], [new_output], name=name, **kwargs), self)
        for i, n in enumerate(node.input):
            if n == input_name:
                node.input[i] = new_output
                break
        return new_node

    def insert_new_node_on_output(self, op_type, output_name, name=None, **kwargs):
        """Create and insert a new node into the graph.
        Args:
            op_type: type for new operation
            output_name: the names of the outputs above us
            name: the name of the new op
            kwargs: attributes of the new node

        Returns:
            node that was inserted
        """
        assert isinstance(output_name, str) and isinstance(op_type, str)
        new_output = name + ":0"
        new_node = Node(helper.make_node(op_type, [output_name], [new_output], name=name, **kwargs), self)
        self.replace_all_inputs(self.get_nodes(), output_name, new_output)
        return new_node

    def find_output_consumers(self, output_name):
        """Find all nodes consuming a given output."""
        nodes = []
        for node in self.get_nodes():
            if output_name in node.input:
                nodes.append(node)
        return nodes

    @staticmethod
    def replace_all_inputs(ops, old_input, new_input):
        """Replace all inputs pointing to old_input with new_input."""
        for node in ops:
            for i, input_name in enumerate(node.input):
                if input_name == old_input:
                    node.input[i] = new_input

    @staticmethod
    def replace_input(node, old, new):
        """Replace node."""
        assert isinstance(node, Node) and isinstance(old, str) and isinstance(new, str)
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
                return True
        return False

    @staticmethod
    def replace_subgraph(ops, subgraph_nodes, old_inputs, old_outputs, new_inputs, new_outputs):
        """Replace subgraph."""
        if len(old_inputs) != len(new_inputs) or len(old_outputs) != len(new_outputs):
            raise ValueError("replace_subgraph - inputs and outputs need to be same length")

        # point all children nodes inputs to the new node
        for oo, no in zip(old_outputs, new_outputs):
            for output_name in oo.output:
                for child in ops:
                    for i, name in enumerate(child.input):
                        if name == output_name:
                            child.input[i] = no.name + ":0"

        # delete nodes no longer used
        removed = set()
        for node in subgraph_nodes.get_nodes():
            if not node or node in removed:
                continue
            ops.remove(node)
            removed.add(node)
        ops.extend(new_outputs)
        return ops

    @staticmethod
    def remove_deleted_nodes(ops):
        return [node for node in ops if not node.is_deleted()]
