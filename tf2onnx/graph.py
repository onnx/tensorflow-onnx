# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.graph - class to manage graph manipulation on top of onnx
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import numpy as np
import six
from onnx import helper, numpy_helper, optimizer, OperatorSetIdProto, AttributeProto

from tf2onnx import utils, __version__
from tf2onnx.utils import port_name, find_opset


class Node(object):
    """A Node - wrapper around onnx nodes that we use for graph manipulations."""

    def __init__(self, node, graph, skip_conversion=False):
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

        graph.set_node_by_name(self)
        # dict to original attributes
        for a in node.attribute:
            self._attr[a.name] = a
        # try to find a dtype for this node
        dtype = graph.get_dtype(node.name)
        if not dtype:
            dtype = self._attr.get("dtype")
            if dtype:
                dtype = dtype.i
        self._dtype = dtype
        self.data_format = self.get_attr("data_format")
        if self.data_format:
            self.data_format = self.data_format.s.decode("utf-8")
        self._skip_conversion = skip_conversion

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def inputs(self):
        """Input node objects."""
        val = [self.graph.get_node_by_output(n) for n in self._input]
        return val

    @property
    def attr(self):
        return self._attr

    @property
    def attr_onnx(self):
        onnx_attrs = {}
        for a in self._attr.values():
            if a.name in utils.ONNX_VALID_ATTRIBUTES:
                onnx_attrs[a.name] = a
        return onnx_attrs

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

    @domain.setter
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

    def get_attr(self, name, default=None):
        """Get attribute map."""
        attr = self.attr.get(name, default)
        return attr

    def get_attr_int(self, name):
        """Get attribute map."""
        attr = self.attr.get(name)
        if attr is not None:
            attr = attr.i
        else:
            attr = 0
        return attr

    def set_attr(self, name, value):
        self.attr[name] = helper.make_attribute(name, value)

    def set_attr_onnx(self, value):
        self.attr[value.name] = value

    def set_deleted(self):
        self.type = "@@DELETED@@"

    def is_deleted(self):
        return self.type == "@@DELETED@@"

    # If some Node is created as onnx_node, then we don't need convert it
    def need_skip(self):
        return self._skip_conversion

    @property
    def output_shapes(self):
        """Get output shapes, mostly for debugging."""
        val = [self.graph.get_shape(n) for n in self._output]
        return val

    @property
    def output_dtypes(self):
        """Get output dtypes, mostly for debugging."""
        val = [self.graph.get_dtype(n) for n in self._output]
        return val

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
            if not t.dims:
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
        for i, _ in enumerate(t.dims):
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

    def get_implicit_inputs(self, require_input_in_cur_graph=False):
        """Get implicit inputs if the node has attributes being GraphProto."""
        body_graphs = [attr.g for attr in self.op.attribute if attr.g]
        outer_scope_node_input_ids = set()
        for sub_g in body_graphs:
            outer_scope_node_input_ids |= self._get_implicit_inputs(sub_g)

        if require_input_in_cur_graph:
            # only find referenced node in current graph
            implicit_inputs_in_current_graph = set()
            for input_id in outer_scope_node_input_ids:
                n = self.graph.get_node_by_output(input_id)
                if n is None:
                    if not self.graph.is_initializer(input_id):
                        if input_id not in self.graph.model_inputs:
                            continue
                implicit_inputs_in_current_graph.add(input_id)
            return implicit_inputs_in_current_graph
        return outer_scope_node_input_ids

    @staticmethod
    def _get_implicit_inputs(onnx_graph, recursive=True):
        """Get implicit inputs for specified onnx graph."""
        node_map = set()
        for n in onnx_graph.node:
            node_map |= set(n.output)

        for n in onnx_graph.input:
            node_map.add(n.name)

        outer_scope_node_input_ids = set()
        for n in onnx_graph.node:
            for i in n.input:
                if i not in node_map:
                    outer_scope_node_input_ids.add(i)

        if recursive:
            for n in onnx_graph.node:
                for attr in n.attribute:
                    sub_g = attr.g
                    if sub_g:
                        outer_scope_node_input_ids |= Node._get_implicit_inputs(sub_g)

        return outer_scope_node_input_ids


class Graph(object):
    """"Class that provides graph manipulation and matching."""

    def __init__(self, nodes, output_shapes=None, dtypes=None, target=None, opset=None, extra_opset=None,
                 output_names=None):
        """Create Graph.
        Args:
            nodes: list of Node()
            output_shapes: dict of tensorflow output shapes
            dtypes: dict of tensorflow dtype
        """
        if target is None:
            target = []
        self._nodes = []
        self._initializers = {}
        self._nodes_by_name = {}
        self._output_to_node_name = {}
        self.shapes = {}
        self._model_inputs = {}
        self._target = set(target)
        self._dtypes = dtypes

        self._output_shapes = output_shapes
        ops = [Node(node, self) for node in nodes]
        self.set_nodes(ops)
        self._opset = find_opset(opset)
        self._extra_opset = extra_opset
        self.output_names = output_names

    @property
    def opset(self):
        return self._opset

    @property
    def initializers(self):
        return self._initializers

    @property
    def model_inputs(self):
        return self._model_inputs

    def is_target(self, name):
        """Return True if target platform is name."""
        return name in self._target

    def is_initializer(self, name):
        """Check the name is a constant value. name is in format - node_name:<int>."""
        return name in self._initializers

    def set_initializer(self, name, val):
        """Set initializer."""
        self._initializers[name] = val

    def make_const(self, name, np_val, skip_conversion=False):
        """Make a new constant in the graph"""
        onnx_tensor = numpy_helper.from_array(np_val, name)
        self.add_initializer(onnx_tensor)
        node = Node(helper.make_node("Const", [], [name], name=name, value=onnx_tensor), self, skip_conversion)
        return node

    def make_node(self, op_type, inputs, attr=None, output_count=1, outputs=None, skip_conversion=True,
                  op_name_scope=None, name=None, shapes=None, dtypes=None):
        """Make a new onnx node in the graph"""
        if attr is None:
            attr = {}
        if shapes is None:
            shapes = []
        if dtypes is None:
            dtypes = []

        op_name_basis = op_type
        if op_name_scope:
            op_name_basis = "_".join([op_name_scope, op_type])

        if name is None:
            name = utils.make_name(op_name_basis)

        if outputs is None:
            outputs = [name + ":" + str(i) for i in range(output_count)]

        raw_attr = {}
        onnx_attrs = []
        for a, v in attr.items():
            if isinstance(v, AttributeProto):
                onnx_attrs.append(v)
            else:
                raw_attr[a] = v
        onnx_node = helper.make_node(op_type, inputs, outputs, name=name, **raw_attr)
        node = Node(onnx_node, self, skip_conversion=skip_conversion)
        if onnx_attrs:
            _ = [node.set_attr_onnx(a) for a in onnx_attrs]

        if shapes:
            utils.make_sure(len(shapes) == output_count, "output shape count not equal to output count")
            for i in range(output_count):
                self.set_shape(node.output[i], shapes[i])

        if dtypes:
            utils.make_sure(len(dtypes) == output_count, "output dtypes count not equal to output count")
            for i in range(output_count):
                self.set_dtype(node.output[i], dtypes[i])

        return node

    def set_nodes(self, ops):
        """Set new node list."""
        self._nodes = ops
        self._nodes_by_name = {op.name: op for op in ops}
        self._output_to_node_name = {}
        for op in ops:
            for op_output in op.output:
                self._output_to_node_name[op_output] = op.name

    def update_proto(self):
        """Update the onnx protobuf from out internal Node structure."""
        for node in self._nodes:
            node.update_proto()

        # update attributes to proto
        for op in self.get_nodes():
            onnx_op = op.op
            del onnx_op.attribute[:]
            attr = [a for a in op.attr_onnx.values()]
            if attr:
                onnx_op.attribute.extend(attr)

    def get_nodes(self):
        """Get node list."""
        return self._nodes

    def get_node_by_output(self, output):
        """Get node by node output id"""
        name = self._output_to_node_name.get(output)
        ret = None
        if name:
            ret = self._nodes_by_name.get(name)
        else:
            ret = self._get_initializer_as_const_node(output)

        return ret

    def get_node_by_name(self, name):
        """Get node by name."""
        ret = self._nodes_by_name.get(name)
        if not ret:
            ret = self._get_initializer_as_const_node(name)
        return ret

    def _get_initializer_as_const_node(self, name):
        """Create dummy const node representing initializers for easier node manipulation"""
        ret = None
        # if we processed the graph fully, set_nodes() the graph has no longer const nodes
        # since we moved them to be initializers. But all graph processing code uses Node
        # as the common data structure. To avoid special casing lots of code for initializers
        # we create a dummy 'Const' Node here.
        initializer = self._initializers.get(name)
        if initializer is not None:
            ret = self.make_node("Const", inputs=[], outputs=[name], name=name,
                                 attr={"value": initializer})
        return ret

    def set_node_by_name(self, node):
        """Set node by name."""
        self._nodes_by_name[node.name] = node
        for op_output in node.output:
            self._output_to_node_name[op_output] = node.name

    def add_model_input(self, name, tensor_value_info):
        """Add placeholder node as model's input"""
        if name not in self._model_inputs:
            self._model_inputs[name] = tensor_value_info
        else:
            raise ValueError("model input already exists")

    def add_initializer(self, tensor):
        """Add tensor to initializers."""
        self._initializers[tensor.name] = tensor
        self.set_shape(tensor.name, tensor.dims)

    def get_initializer(self, name):
        """Return tensor or throw exception if it does not exist."""
        if self.is_initializer(name):
            return self._initializers[name]
        raise ValueError("no initializer called " + name)

    def update_initializer(self, name, tensor):
        if self.is_initializer(name):
            new_tensor = numpy_helper.from_array(tensor, name)
            if new_tensor.dims != self._initializers[name].dims:
                self.set_shape(name, new_tensor.dims)

            del self._initializers[name]
            self._initializers[name] = new_tensor
        else:
            raise ValueError("no initializer called " + name)

    def get_dtype(self, name):
        """Get dtype for node."""
        return self._dtypes.get(name)

    def set_dtype(self, name, dtype):
        """Set dtype for node."""
        self._dtypes[name] = dtype

    def get_shape(self, name):
        """Get shape for node."""
        assert isinstance(name, six.text_type)
        shape = self._output_shapes.get(name)
        if shape:
            for i, v in enumerate(shape):
                if v is None:
                    shape[i] = -1
            # hack to allow utils.ONNX_UNKNOWN_DIMENSION to override batchsize if needed.
            # default is -1.
            if shape[0] == -1:
                shape[0] = utils.ONNX_UNKNOWN_DIMENSION
        return shape

    def set_shape(self, name, val):
        """Set new shape of node."""
        if isinstance(val, np.ndarray):
            val = val.tolist()
        self._output_shapes[name] = val

    def copy_shape(self, input_name, output_name):
        """Copy shape from another node."""
        shape = self.get_shape(input_name)
        # assert shape is not None
        if shape is not None:
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
            all_input = set(op.input)
            implicit_inputs = op.get_implicit_inputs()
            all_input |= set(implicit_inputs)
            for inp in all_input:
                j = self.get_node_by_output(inp)
                if j and j.type != "Const":
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
            while stack:
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

    def make_graph(self, doc, graph_name="tf2onnx"):
        """
        Create GraphProto for onnx from internal graph.
        Args:
            optimize: optimize graph via onnx
            doc: text for doc string of the model
        """
        self.update_proto()

        # TODO: we'd want to do something like this so that transpose optimizer is active
        # for  all (unit) tests
        # if optimize:
        #    from tf2onnx.optimizer.transpose_optimizer import TransposeOptimizer
        #    optimizer = TransposeOptimizer(self, False)
        #    optimizer.optimize()

        # create output_tensor_values
        output_tensor_values = []
        for name in self.output_names:
            dtype = self.get_dtype(name)
            if not dtype:
                raise ValueError("cannot found the output dtype for " + name)
            v = helper.make_tensor_value_info(name, dtype, utils.make_onnx_shape(self.get_shape(name)))
            output_tensor_values.append(v)

        # update attributes
        ops = []
        all_inputs = set()
        for op in self.get_nodes():
            all_inputs |= set(op.input)
            all_inputs |= set(op.get_implicit_inputs())
            onnx_op = op.op
            ops.append(onnx_op)

        # create input_tensor_values, initializers
        # if initializer is not used as input by any node, then it will be ignored
        initializers = [i for i in list(self._initializers.values()) if i.name in all_inputs]
        input_with_initializers = []
        for initializer in initializers:
            shape = self.get_shape(initializer.name)
            if shape and list(shape) != initializer.dims:
                raise ValueError("initializer shape is inconsistent for " + initializer.name)
            val = helper.make_tensor_value_info(initializer.name,
                                                initializer.data_type,
                                                utils.make_onnx_shape(initializer.dims))
            input_with_initializers.append(val)

        input_with_initializers.extend(list(self._model_inputs.values()))

        # create model proto
        graph = helper.make_graph(ops, graph_name,
                                  input_with_initializers,
                                  output_tensor_values,
                                  initializer=initializers,
                                  doc_string=doc)

        return graph

    def make_model(self, doc, optimize=False, graph_name="tf2onnx"):
        """
        Create final ModelProto for onnx from internal graph.
        Args:
            optimize: optimize graph via onnx
            doc: text for doc string of the model
            output_names: list of model outputs
        """
        graph = self.make_graph(doc, graph_name)
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

        # optimize the model proto.
        # TODO: this is disabled by default because of bugs in fuse_consecutive_transposes
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
        val.append("{}{} {} {}".format(space, node.type, node.name, self.get_shape(port_name(node.name))))
        space += "    "
        for j in node.inputs:
            val.extend(self.follow_inputs(j, num - 1, space))
        if top:
            print("\n".join(reversed(val)))
            print()
            return []
        return val

    def dump_node_statistics(self):
        op_cnt = collections.Counter()
        for n in self.get_nodes():
            op_cnt[n.type] += 1

        return op_cnt

    @staticmethod
    def remove_input(node, to_be_removed):
        """Remove input from Node.
        Args:
            node: the node we expect the input on
            to_be_removed: the node name we want to remove
        """
        assert isinstance(node, Node) and isinstance(to_be_removed, six.text_type)
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
        new_output = port_name(name)
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
        assert isinstance(output_name, six.text_type) and isinstance(op_type, six.text_type)
        new_output = port_name(name)
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
    def replace_input(node, old_input, new_input):
        """Replace node."""
        assert isinstance(node, Node) and isinstance(old_input, six.text_type) and isinstance(new_input, six.text_type)
        is_replaced = False
        for i, input_name in enumerate(node.input):
            if input_name == old_input:
                node.input[i] = new_input
                is_replaced = True
        return is_replaced

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
                            child.input[i] = port_name(no.name)

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

    @staticmethod
    def _extract_sub_graph_nodes(dest_node):
        """
        return related nodes of specified node
        :param dest_node: the specified node
        :return: set of related nodes
        """
        res_set = set()
        processing_set = set([dest_node])
        while processing_set:
            top_node = processing_set.pop()
            res_set.add(top_node)
            for node in top_node.inputs:
                if not node:
                    # some node (for example Scan) has optional inputs, which
                    # might has empty input.
                    continue
                if node not in res_set:
                    processing_set.add(node)
        return res_set

    def extract_sub_graph_nodes(self, outputs_name):
        """
        find related nodes of specified list of nodes
        """
        res_set = set()
        for output in outputs_name:
            node = self.get_node_by_output(output)
            res_set = res_set.union(self._extract_sub_graph_nodes(node))

        # const nodes made at conversion stage are not needed, because ONNX will use initializer automatically
        not_need_const = set()
        for node in res_set:
            if self.is_initializer(node.name):
                not_need_const.add(node)

        res_set = res_set - not_need_const
        return list(res_set)

    def delete_unused_nodes(self, outputs_name):
        """
        example of unused nodes: loop nodes generated by tensorflow rnn such as switch, loop_cond
        """
        related_nodes = self.extract_sub_graph_nodes(outputs_name)
        self.set_nodes(related_nodes)
