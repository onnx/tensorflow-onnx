import collections
import numpy as np
import onnx
from onnx import helper, numpy_helper
from tf2onnx.optimizer.onnx_graph import OnnxGraph, OnnxNode
from tf2onnx import utils

def is_nhwc_transpose(transpose_node):
    return transpose_node.type == "Transpose" and transpose_node.get_attr('perm').ints == [0, 2, 3, 1]

def is_nchw_transpose(transpose_node):
    return transpose_node.type == "Transpose" and transpose_node.get_attr('perm').ints == [0, 3, 1, 2]

def is_useless_transpose(transpose_node):
    return transpose_node.type == "Transpose" and transpose_node.get_attr('perm').ints == [0, 1, 2, 3]

def single_output_node_check(g, node_names):
    result = True
    for n in node_names:
        result = result and g.has_single_output_node(n)
        if not result:
            return False
    return True

def make_onnx_node(g, operation_type, input_names_with_output_id, attribute = None, output_num = 1):
    op_name = utils.make_name(operation_type)
    out_names = []
    for i in range(output_num):
        out_names.append(op_name + ":" + str(i))

    n = helper.make_node(operation_type, input_names_with_output_id, out_names, name=op_name)
    if attribute:
        n.attribute.extend(attribute)

    return OnnxNode(n, g)


def get_non_nchw_transpose_output_nodes(node, total_output_cnt, quit_once_found_once = False):
    assert len(node.output) == 1 # we just support node having 1 output
    non_nchw_tranpose_nodes = []
    consumers = node.get_output_consumers_at_index(0)
    total_output_cnt.clear()
    total_output_cnt.append(len(consumers))
    for o in consumers:
        if not is_nchw_transpose(o) and o not in non_nchw_tranpose_nodes:
            non_nchw_tranpose_nodes.append(o)
            if quit_once_found_once:
                break
    return non_nchw_tranpose_nodes

#transpose_node_cnt means how many tranpose can be eliminated in the node's input
def create_transpose_pairs_before_node(g, node, transpose_node_cnt):
    is_log = False
    assert len(node.output) == 1 # just support node who has 1 output
    output_cnt = []
    non_nchw_trans_consumers = get_non_nchw_transpose_output_nodes(node, output_cnt)

    if len(non_nchw_trans_consumers) == 0:
        return 1

    # comment those in case of later use.
    # currently we will definitely add transpose ops to make the Transpose goes down
    #total_output_node_cnt = output_cnt[0]
    #eliminated_num = (total_output_node_cnt - len(non_nchw_trans_consumers)) + transpose_node_cnt
    #if len(non_nchw_trans_consumers) > 0 and eliminated_num < len(non_nchw_trans_consumers):
    #    if is_log:
    #        print(node.name + ", " + str(eliminated_num) + str(total_output_node_cnt))
    #        print("it is not reasonable if we add more transpose (than original graph) just to make out pattern match")
    #    return 0

    added_node = []
    # add Transpose(0, 3, 1, 2) and Transpose(0, 2, 3, 1) before each non_nchw_trans_consumers
    for consumer in non_nchw_trans_consumers:
        nchw_op_name = utils.make_name("Transpose")
        nchw_out_name = nchw_op_name + ":0"

        kwargs = {"perm": [0, 3, 1, 2]}
        nchw = helper.make_node("Transpose", [node.output[0]], [nchw_out_name], name=nchw_op_name, **kwargs)

        nhwc_op_name = utils.make_name("Transpose")
        nhwc_out_name = nhwc_op_name + ":0"

        kwargs = {"perm": [0, 2, 3, 1]}
        nhwc= helper.make_node("Transpose", [nchw_out_name], [nhwc_out_name], name=nhwc_op_name, **kwargs)
        nchw_node = OnnxNode(nchw, g)
        nhwc_node = OnnxNode(nhwc, g)
        consumer.replace_input(node.output[0], nhwc_out_name)

        added_node.extend([nchw_node, nhwc_node])
    ops = g.get_nodes()
    ops.extend(added_node)
    g.set_nodes(ops)
    if is_log:
        print(str(len(added_node)) + " nodes are added")
        print(added_node)
    return 2

def are_all_input_nodes_nhwc(node):
    all_true = True
    for n in node.inputs:
        if not is_nhwc_transpose(n):
            all_true = False
            break
    return all_true

def handle_node_having_branches(g, p):
    is_log = False
    # make sure p's all input all have only 1 output, otherwise, it would impact their other outputs
    if are_all_input_nodes_nhwc(p) and single_output_node_check(g, p.input):
        #print("handle_node_having_branches" + p.name)
        status_code = create_transpose_pairs_before_node(g, p, len(p.input))
        if is_log:
            print("status code ====> " + str(status_code) + "")
        if status_code != 0:
            ops = g.get_nodes()
            to_remove = []

            input_transposes = p.inputs
            for n in input_transposes:
                n_input = n.input[0]
                n_outputs = n.get_output_consumers_at_index(0)
                for m in n_outputs:
                    m.replace_input(n.output[0], n_input)

                if is_log:
                    print("[input] replace " + n.name + " with " + n_input.name)
                to_remove.append(n)

            output_transposes = p.get_output_consumers_at_index(0)
            for n in output_transposes:
                if is_log:
                    print("[output] replace " + n.name + " with " + p.name)
                n_input = n.input[0]
                n_outputs = n.get_output_consumers_at_index(0)
                for m in n_outputs:
                    m.replace_input(n.output[0], n_input)
                to_remove.append(n)

            [ops.remove(n) for n in to_remove]
            g.set_nodes(ops)
            return True
        else:
            #print("not reasonable to create nchw node to match deletion pattern")
            pass

    else:
        #print("not all input path ends with nhwc transpose, skipping...")
        pass

# the assumption is: only node.input[0] and trans.input[0] will be token care here.
# if node has other input, they should be const
def switch_transpose_and_node(g, node, trans):
    ops = g.get_nodes()
    g.replace_subgraph_output(ops, node, trans)
    node.input[0] = trans.input[0]
    trans.input[0] = node.name + ":0"
    g.set_nodes(ops)

def handle_hnwc_tranpose(g, trans, force_stop):
    #print("@@@@@@Transpose " + trans.name)
    out_nodes = trans.get_output_consumers_at_index(0)
    if len(out_nodes) == 1:
        first_out = out_nodes[0]
        p = first_out

        to_remove = []
        to_extend = []
        if p.type == "Add":
            if g.is_initializer(p.input[1]):
                t_p = trans.inputs[0]
                if t_p.type == "Conv" and len(t_p.input) == 2:
                    # if Conv's bias input is not set, then we set, otherwise, we don't set
                    # todo: maybe we can add already set bias with the input??? try later
                    conv_node = make_onnx_node(g, "Conv", [t_p.input[0], t_p.input[1], p.input[1]], t_p.op.attribute)

                    ops = g.get_nodes()
                    trans.input[0] = conv_node.name + ":0"
                    g.replace_subgraph_output(ops, p, trans)
                    ops.remove(t_p)
                    ops.remove(p)
                    ops.append(conv_node)
                    g.set_nodes(ops)
                    return True
                else:
                    print("shift add.input[1] to left")
            else:
                status = handle_node_having_branches(g, p)
                if status == True:
                    return True

        elif p.type == "Relu":
            ops = g.get_nodes()
            cnt = g.replace_subgraph_output(ops, p, trans)
            p.input[0] = trans.input[0]
            trans.input[0] = p.name + ":0"
            g.set_nodes(ops)
            return True
        elif is_nchw_transpose(p):
            ops = g.get_nodes()
            g.replace_subgraph_output(ops, p, trans.inputs[0])
            ops.remove(p)
            ops.remove(trans)
            g.set_nodes(ops)
            return True
        elif p.type in ["Max", "Min"]:
            assert g.is_initializer(p.input[1])
            numpy_val = numpy_helper.to_array(g.get_initializer_tensor(p.input[1]))
            transposed_val = np.transpose(numpy_val, (0, 3, 1, 2))
            g.update_initializer_tensor(p.input[1], transposed_val)
            switch_transpose_and_node(g, p, trans)
            return True
        elif p.type == "Mul":
            mul = p
            # make sure conv don't have bias set
            if g.is_initializer(mul.input[1]):
                t_p = trans.inputs[0]
                if t_p.type == "Conv" and g.is_initializer(t_p.input[1]) and len(t_p.input) == 2:
                    #print("Fuse Conv's weight with susquent constant multiply")
                    conv = t_p
                    numpy_val = numpy_helper.to_array(g.get_initializer_tensor(conv.input[1]))
                    transposed_val = np.transpose(numpy_val, (2, 3, 1, 0))
                    mul_val = numpy_helper.to_array(g.get_initializer_tensor(mul.input[1]))
                    result = np.multiply(transposed_val, mul_val)
                    g.update_initializer_tensor(conv.input[1], np.transpose(result, (3, 2, 0, 1)))

                    ops = g.get_nodes()
                    g.replace_subgraph_output(ops, p, trans)
                    ops.remove(p)
                    g.set_nodes(ops)
                    return True
                else:
                    muler_dim = g.get_initializer_tensor(mul.input[1]).dims[0]
                    if muler_dim == 1: # if there is only 1 number, so we just move trasponse after the mul
                        ops = g.get_nodes()
                        cnt = g.replace_subgraph_output(ops, p, trans)

                        p.input[0] = trans.input[0]
                        trans.input[0] = p.name + ":0"
                        g.set_nodes(ops)
                        return True
                    else: # if the muler is not a single number, we need shift the data
                        print("shift Conv's weight to NCHW....")
            else:
                print("Mul's second input is not a const, skipping")
                pass
        elif p.type == "Identity":
            ops = g.get_nodes()
            g.replace_subgraph_output(ops, p, trans)
            ops.remove(p)
            g.set_nodes(ops)
            return True
        elif p.type == "Concat":
            if handle_node_having_branches(g, p) == True:
                p.set_attr("axis", 1)
                return True
        elif p.type == "Split":
            # Todo: need handle cases where Slit node has more than 1 outputs.
            if handle_node_having_branches(g, p) == True:
                p.set_attr("axis", 1)
                return True

        elif p.type == "Pad":
            pad = p
            #[N-start, H-start,  W-start, C-start, N-end, H-end,  W-end, C-end]
            pads = pad.get_attr('pads').ints #[x1_begin, x2_begin...x1_end, x2_end,...]
            # NHWC->NCHW
            new_pads = [pads[0], pads[3], pads[1], pads[2], pads[4], pads[7], pads[5], pads[6] ]
            p.set_attr("pads", new_pads)
            switch_transpose_and_node(g, pad, trans)
            return True

        elif p.type == "ReduceMean":
            axes = p.get_attr("axes").ints
            keepdims = p.get_attr("keepdims")
            # make sure keepdims is 1, then we can do the swap, otherwise, plese don't, because
            # once keepdims is not set, original dims are lost, so transpose back won't work well.
            # by default, if keepdims is not specified, it is 1
            if axes == [1, 2] and ((keepdims and keepdims.i == 1) or (not keepdims)):
                p.set_attr("axes", [2, 3])
            else:
                return
            switch_transpose_and_node(g, p, trans)
            return True

        elif p.type == "Slice":
            axes = p.get_attr("axes").ints
            keepdims = p.get_attr("keepdims")
            if axes == [0, 1, 2, 3]:
                p.set_attr("axes", [0, 2, 3, 1])
            else:
                return
            switch_transpose_and_node(g, p, trans)
            return True
    else:
        # move transpose into branches to let more pattern match
        to_append = []
        for n in out_nodes:
            branch_trans = make_onnx_node(g, "Transpose", [trans.input[0]], trans.op.attribute, 1)
            n.replace_input(trans.output[0], branch_trans.output[0])
            to_append.append(branch_trans)
        ops = g.get_nodes()
        ops.remove(trans)
        ops.extend(to_append)
        g.set_nodes(ops)
        #print("move transpose into branches to let more pattern match, transpose name: " + trans.name)
        pass

def remove_useless_tranpose(g, trans):
    #print("remove useless transpose" + trans.name + ", previous node name " + trans.inputs[0].name + " of type " + str(trans.inputs[0].type))
    ops = g.get_nodes()
    # some Transpose node has parent node being BatchNormalization, whos has more than 1 output
    outputs = trans.get_output_consumers_at_index(0)
    for n in outputs:
        n.replace_input(trans.output[0], trans.input[0])
    ops.remove(trans)
    g.set_nodes(ops)

class TransposeOptimizer(object):
    def __init__(self, graph):
        self.g = graph

        self.pre_optimize_action()

    @property
    def nodes(self):
        return self.g.get_nodes()

    def pre_optimize_action(self):
        # make Reshape into a const, which then can be fused into Conv's weight for mobilenet_v1_75_192
        ops = self.nodes
        constable_reshape_ops = [n for n in ops if (n.type == "Reshape" and self.g.is_initializer(n.input[0]) and self.g.is_initializer(n.input[1]))]
        for reshape_op in constable_reshape_ops:
            target_t = numpy_helper.to_array(self.g.get_initializer_tensor(reshape_op.input[0]))
            target_shape = numpy_helper.to_array(self.g.get_initializer_tensor(reshape_op.input[1]))
            new_data = np.reshape(target_t, tuple(target_shape))
            const_name = utils.make_name("Const") + ":0"
            new_tensor = numpy_helper.from_array(new_data, const_name)

            # point all children nodes inputs to the new node
            for output_name in reshape_op.output:
                for child in ops:
                    for i, name in enumerate(child.input):
                        if name == output_name:
                            child.input[i] = const_name
            self.g.add_initializer(new_tensor)
            # need call this to make input update synced to protobuf val
            self.g.update_proto()
            ops.remove(reshape_op)
            self.g.set_nodes(ops)
            self.g.topological_sort(ops)

    def optimize(self):
        self.g.calculate_node("before optimization")
        no_action = False
        iteration_cnt = 0
        while(not no_action):
            no_action = True
            nodes = self.nodes
            force_stop = {}
            for n in nodes:
                if is_nhwc_transpose(n):
                    if handle_hnwc_tranpose(self.g, n, force_stop):
                        no_action = False
                        iteration_cnt += 1
                        #print("####Break after some action")
                        # need break, because handler may change nodes set, making the n stale object referencing already deleted elements
                        break
                    
                if is_useless_transpose(n):
                    no_action = False
                    iteration_cnt += 1
                    remove_useless_tranpose(self.g, n)
                    #print("####Break after some action 2")
                    break
            if "1" in force_stop and force_stop["1"] == 1:
                break
        self.g.calculate_node("after optimization")
        print("finish after " + str(iteration_cnt) + " iteration(s)")
        self.g.update_proto()
        model_proto = self.g.make_model(optimize=False)

        return model_proto