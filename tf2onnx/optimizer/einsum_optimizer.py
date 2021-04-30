# SPDX-License-Identifier: Apache-2.0

"""Rewrites operator einsum into simple ONNX operators.
"""

from __future__ import unicode_literals
import numpy as np
from onnx import helper, numpy_helper
from onnx.defs import onnx_opset_version
from onnx.onnx_pb import TensorProto
from ..constants import OPSET_TO_IR_VERSION


def single_axes(axes):
    """
    *axes* contains positive values, then it is the position
    of this axis in the original matrix, otherwise it is -1
    meaning this axis is an added single dimension to align
    all the dimensions based on the einsum equation.

    :param axes: axes described above
    :return: list of integer in set `{1, 2}`, 1 for
        a single axis, 2 otherwise
    """
    if axes is None:
        return axes
    return [(1 if a == -1 else 2) for a in axes]


class EinsumSubOp:
    """
    Defines a sub operation used in Einsum decomposition.

    :param name: name (reshape, transpose, reduce_sum, matmul, id,
        squeeze, diagonal, mul, batch_dot)
    :param inputs: inputs
    :param kwargs: arguments

    Operator suffixed by `_mm` (*transpose_mm*, *reduce_sum_mm*)
    are equivalent to the same operator without the suffix
    but takes two inputs and only changes the first one.
    """
    _allowed = {'expand_dims', 'transpose', 'reduce_sum', 'matmul', 'id',
                'squeeze', 'diagonal', 'mul', 'batch_dot',
                'transpose_mm', 'reduce_sum_mm'}

    def __init__(self, full_dim, name, *inputs, **kwargs):
        self.full_dim = full_dim
        self.name = name
        self.inputs = inputs
        self.kwargs = kwargs
        self._info = {}
        if name not in EinsumSubOp._allowed:
            raise ValueError(
                "Unexpected name %r. It should be in %r."
                "" % (name, EinsumSubOp._allowed))
        if len(inputs) not in (1, 2):
            raise RuntimeError(
                "Inputs must contains 1 or 2 inputs not %d." % len(inputs))
        if name == 'matmul' and len(inputs) != 2:
            raise RuntimeError(
                "Inputs must contains 2 inputs not %d for operator 'matmul'."
                "" % len(inputs))
        for i, inp in enumerate(inputs):
            if not isinstance(inp, (int, EinsumSubOp)):
                raise TypeError(
                    "Input %d has type %r, int or EinsumSubOp is expected."
                    "" % (i, type(inp)))
        self._check_()

    def _check_(self):
        if self.name == 'transpose':
            self._check_arg_('perm', tuple)
            perm = self.kwargs['perm']
            if len(perm) != len(set(perm)):
                raise RuntimeError(
                    "perm has duplicated values %r (name=%r)."
                    "" % (perm, self.name))
            if list(perm) == list(range(len(perm))):
                raise ValueError(
                    "Transpose = identity perm=%r. It must be removed."
                    "" % perm)
        elif self.name == 'matmul':
            self._check_arg_('axes', tuple)
            self._check_arg_('left', tuple)
            self._check_arg_('right', tuple)
            axes = self.kwargs['axes']
            left = self.kwargs['left']
            right = self.kwargs['right']
            for a in axes:
                if a in left and a in right:
                    raise RuntimeError(
                        "One axis belongs to every set (axes, left, right). "
                        "axes=%r, left=%r, right=%r." % (axes, left, right))

    def __repr__(self):
        inps = ", ".join(map(str, self.inputs))
        kw = ", ".join("%s=%r" % (k, w) for k, w in self.kwargs.items())
        m = "%s(%r, %s, %s)" % (
            self.__class__.__name__, self.name, inps, kw)
        return m

    def _check_arg_(self, name, typ, empty=False):
        if name not in self.kwargs:
            raise RuntimeError(
                "Parameter %r not found for operator %r." % (name, self.name))
        if empty and self.kwargs[name] is None:
            return
        if not isinstance(self.kwargs[name], typ):
            raise TypeError(
                "Unexpected type %r for parameter %r and parameter %r."
                "" % (type(self.kwargs[name]), name, self.name))

    def _check_row_(self, row, inp=False):
        """
        Checks input or output is valid.
        """
        pass

    def _compute_output_row_id(self, row, row2=None, ab=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        row[:] = row2[:]
        self._check_row_(row)

    def _compute_output_row_transpose(self, row, row2=None, ab=False):
        if ab:
            self._compute_output_row_transpose(row2)
            return
        self._check_row_(row, True)
        self._check_arg_('perm', tuple)
        if len(self.kwargs['perm']) != len(row):
            raise RuntimeError(
                "Unexpected permutation %r (row=%r)."
                "" % (self.kwargs['perm'], row))
        perm = self.kwargs['perm']
        cpy = row.copy()
        for i, p in enumerate(perm):
            row[i] = cpy[p]
        self._check_row_(row)

    def _compute_output_row_transpose_mm(self, row, row2=None, ab=False):
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True)
        if row2 is None:
            raise RuntimeError("transpose_mm expects a second input.")
        self._compute_output_row_transpose(row, row2=None)

    def _compute_output_row_expand_dims(self, row, row2=None, ab=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        self._check_arg_('axes', tuple)
        axes = self.kwargs['axes']
        for axis in axes:
            if not isinstance(axis, tuple):
                raise TypeError(
                    "Parameter axes of expand_dims should be a tuple of "
                    "tuple, axes=%r." % axes)
            if row[axis[1]] != -1:
                raise RuntimeError(
                    "Dimension should be -1 in row %r axis=%r." % (
                        row, self.kwargs['axis']))
        self._check_row_(row)

    def _compute_output_row_reduce_sum(self, row, row2=None, ab=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        self._check_arg_('axes', tuple)
        for a in self.kwargs['axes']:
            row[a] = -1
        self._check_row_(row)

    def _compute_output_row_reduce_sum_mm(self, row, row2=None, ab=False):
        if not ab:
            raise RuntimeError("ab must be true.")
        self._check_row_(row2, True)
        if row2 is None:
            raise RuntimeError("reduce_sum_mm expects a second input.")
        self._compute_output_row_reduce_sum(row, row2=None)

    def _compute_output_row_squeeze(self, row, row2=None, ab=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        self._check_arg_('axes', tuple)
        for a in self.kwargs['axes']:
            row[a] = -1
        self._check_row_(row)

    def _compute_output_row_diagonal(self, row, row2=None, ab=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True)
        self._check_arg_('diag', list)
        to_remove = []
        for choice, choices in self.kwargs['diag']:
            for ch in choices:
                if ch != choice:
                    to_remove.append(ch)
            for i in range(len(row)):  # pylint: disable=C0200
                if row[i] in choices:
                    if row[i] != choice:
                        row[i] = choice
        to_remove.sort()
        for r in to_remove:
            for i in range(len(row)):  # pylint: disable=C0200
                if row[i] == r:
                    raise RuntimeError(
                        "Unexpected result r=%r row=%r to_remove=%r "
                        "diag=%r." % (
                            r, row, to_remove, self.kwargs['diag']))
                if row[i] > r:
                    row[i] -= 1
        self._check_row_(row)

    def _compute_output_row_matmul(self, row, row2=None, ab=False):
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True)
        self._check_row_(row2, True)
        self._check_arg_('axes', tuple)
        self._check_arg_('left', tuple)
        self._check_arg_('right', tuple)
        self._check_arg_('ndim', int)
        if row2 is None:
            raise RuntimeError("matmul expects two inputs.")
        row2[:] = np.maximum(row, row2)
        for a in self.kwargs['axes']:
            if a not in self.kwargs['right']:
                row2[a] = -1
        self._check_row_(row2)

    def _compute_output_row_batch_dot(self, row, row2=None, ab=False):
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True)
        self._check_row_(row2, True)
        self._check_arg_('batch_axes', tuple)
        self._check_arg_('keep_axes', tuple, empty=True)
        self._check_arg_('sum_axes', tuple)
        self._check_arg_('left', tuple)
        self._check_arg_('right', tuple)
        self._check_arg_('ndim', int)
        if row2 is None:
            raise RuntimeError("batch_dot expects two inputs.")
        row2[:] = np.maximum(row, row2)
        for a in self.kwargs['sum_axes']:
            if a not in self.kwargs['right']:
                row2[a] = -1
        self._check_row_(row2)

    def _compute_output_row_mul(self, row, row2=None, ab=False):
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True)
        self._check_row_(row2, True)
        if row2 is None:
            raise RuntimeError("mul expects two inputs.")
        row2[:] = np.maximum(row, row2)
        self._check_row_(row2)

    def compute_output_row(self, row, row2=None, ab=False):
        """
        Updates *row* based on the operator.
        """
        method_name = "_compute_output_row_%s" % self.name
        meth = getattr(self, method_name, None)
        if meth is None:
            raise NotImplementedError(
                "compute_output_row not implemented for %r." % self.name)
        self.add_info(i_row=single_axes(row), i_row2=single_axes(row2))
        meth(row, row2=row2, ab=ab)
        self.add_info(o_row=single_axes(row), o_row2=single_axes(row2))

    def add_info(self, **kwargs):
        """
        Adds information to the node.

        :param kwargs: dictionary
        """
        for k, v in kwargs.items():
            if k in self._info:
                raise KeyError(
                    "Key %r already added (operator %r)." % (k, self.name))
            self._info[k] = v

    def _check_inputs_(self, n_expected, check_dim=False):
        if len(self.inputs) != n_expected:
            raise RuntimeError(
                "Number of inputs must be %d not %d for operator %r."
                "" % (n_expected, len(self.inputs), self.name))

    def _check_shape_(self, m):
        if len(m.shape) != self.full_dim:
            raise RuntimeError(
                "Number of dimensions %r is different from expected value "
                "%d." % (m.shape, self.full_dim))

    def _get_data(self, data, key):
        if isinstance(key, int):
            if key not in data:
                raise RuntimeError(
                    "Unable to find key %d in %r." % (
                        key, list(sorted(data))))
            return data[key]
        if isinstance(key, EinsumSubOp):
            if id(key) not in data:
                raise RuntimeError(
                    "Unable to find key %d in %r." % (
                        id(key), list(sorted(data))))
            return data[id(key)]
        raise TypeError(
            "Unexpected input type %r." % type(key))

    def _onnx_name(self):
        return 'einsum%d_%s' % (id(self), self.name[:2])

    def _check_onnx_opset_(self, opset, limit):
        if opset is not None and opset < limit:
            raise RuntimeError(
                "Opset (%r) must be >= %r for operator %r."
                "" % (opset, limit, self.name))

    def _to_onnx_id(self, names, opset, **kwargs):
        self._check_inputs_(1)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        yield helper.make_node('Identity', [name], [self._onnx_name()])

    def _to_onnx_expand_dims(self, names, opset, **kwargs):
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs['axes']
        name_axes = name + '_axes'
        yield numpy_helper.from_array(
            np.array([a[1] for a in axes], dtype=np.int64), name=name_axes)
        yield helper.make_node(
            'Unsqueeze', [name, name_axes], [self._onnx_name()])

    def _to_onnx_squeeze(self, names, opset, **kwargs):
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs['axes']
        name_axes = name + '_axes'
        yield numpy_helper.from_array(
            np.array(axes, dtype=np.int64), name=name_axes)
        yield helper.make_node(
            'Squeeze', [name, name_axes], [self._onnx_name()])

    def _to_onnx_transpose(self, names, opset, **kwargs):
        self._check_inputs_(1)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        perm = self.kwargs['perm']
        yield helper.make_node(
            'Transpose', [name], [self._onnx_name()], perm=perm)

    def _to_onnx_reduce_sum(self, names, opset, **kwargs):
        self._check_inputs_(1)
        self._check_onnx_opset_(opset, 11)
        inp = self.inputs[0]
        name = self._get_data(names, inp)
        axes = self.kwargs['axes']
        name_axes = self._onnx_name() + '_axes'
        yield numpy_helper.from_array(
            np.array(axes, dtype=np.int64), name=name_axes)
        yield helper.make_node(
            'ReduceSum', [name, name_axes], [self._onnx_name()], keepdims=1)

    def _to_onnx_mul(self, data, **kwargs):
        self._check_inputs_(2)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        yield helper.make_node('Mul', [m1, m2], [self._onnx_name()])

    def _to_onnx_batch_dot(self, names, opset, **kwargs):  # pylint: disable=R0914
        self._check_inputs_(2)
        self._check_onnx_opset_(opset, 13)
        inp1, inp2 = self.inputs[:2]  # pylint: disable=W0632
        name1 = self._get_data(names, inp1)
        name2 = self._get_data(names, inp2)

        batch_axes = self.kwargs['batch_axes']
        keep_axes = self.kwargs['keep_axes']
        sum_axes = self.kwargs['sum_axes']
        left = self.kwargs['left']
        right = self.kwargs['right']
        root = self._onnx_name()

        name_one = root + "_1"
        name_zero = root + "_0"
        yield numpy_helper.from_array(
            np.array([1], dtype=np.int64), name=name_one)
        yield numpy_helper.from_array(
            np.array([0], dtype=np.int64), name=name_zero)

        name_shape1 = root + "_shape1"
        name_shape2 = root + "_shape2"
        concat_left = []
        concat_right = []
        yield helper.make_node('Shape', [name1], [name_shape1])
        yield helper.make_node('Shape', [name2], [name_shape2])

        if len(batch_axes) > 0:
            name_batch_axes = root + "_batch_axes"
            yield numpy_helper.from_array(
                np.array(batch_axes, dtype=np.int64), name=name_batch_axes)

        if len(sum_axes) > 0:
            name_sum_axes = root + "_sum_axes"
            yield numpy_helper.from_array(
                np.array(sum_axes, dtype=np.int64), name=name_sum_axes)

        # dim0 = int(np.prod([m1.shape[i] for i in batch_axes]))
        # dim0b = int(np.prod([m2.shape[i] for i in batch_axes]))
        if len(batch_axes) > 1:
            name_dim0 = root + "_dim0"
            name_dim0b = root + "_dim0b"
            name_dim0g = name_dim0 + 'g'
            name_dim0bg = name_dim0b + 'g'
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)
            yield helper.make_node(
                'Gather', [name_shape1, name_batch_axes], [name_dim0g])
            yield helper.make_node(
                'Gather', [name_shape2, name_batch_axes], [name_dim0bg])
            yield helper.make_node(
                'ReduceProd', [name_dim0g], [name_dim0], keepdims=1)
            yield helper.make_node(
                'ReduceProd', [name_dim0bg], [name_dim0b], keepdims=1)
        elif len(batch_axes) == 1:
            name_dim0g = root + "_dim0g"
            name_dim0bg = root + "_dim0bg"
            name_dim0 = name_dim0g
            name_dim0b = name_dim0bg
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)
            yield helper.make_node(
                'Gather', [name_shape1, name_batch_axes], [name_dim0g])
            yield helper.make_node(
                'Gather', [name_shape2, name_batch_axes], [name_dim0bg])
        else:
            name_dim0 = name_one
            name_dim0b = name_one
            concat_left.append(name_dim0)
            concat_right.append(name_dim0b)

        # dimb = int(-1 if keep_axes is None else np.prod(
        #     [m1.shape[i] for i in keep_axes]))
        if keep_axes in (-1, None) or len(keep_axes) == 0:
            name_dimb = root + "__1"
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                np.array([-1], dtype=np.int64), name=name_dimb)
        elif len(keep_axes) == 1:
            name_keep_axes = root + "_keep_axes"
            name_dimb = root + "_dimb"
            name_dimbg = name_dimb
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                np.array(keep_axes, dtype=np.int64), name=name_keep_axes)
            yield helper.make_node(
                'Gather', [name_shape1, name_keep_axes], [name_dimbg])
        else:
            name_keep_axes = root + "_keep_axes"
            name_dimb = root + "_dimb"
            name_dimbg = name_dimb + 'g'
            concat_left.append(name_dimb)
            concat_right.append(name_dimb)
            yield numpy_helper.from_array(
                np.array(keep_axes, dtype=np.int64), name=name_keep_axes)
            yield helper.make_node(
                'Gather', [name_shape1, name_keep_axes], [name_dimbg])
            yield helper.make_node(
                'ReduceProd', [name_dimbg], [name_dimb], keepdims=1)

        # dim1 = int(np.prod([m1.shape[i] for i in sum_axes]))
        # dim2 = int(np.prod([m2.shape[i] for i in sum_axes]))

        if len(sum_axes) == 0:
            name_dim1 = name_one
            name_dim2 = name_one
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
        elif len(sum_axes) == 1:
            name_dim1 = root + "_dim1"
            name_dim2 = root + "_dim2"
            name_dim1g = name_dim1
            name_dim2g = name_dim2
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
            yield helper.make_node(
                'Gather', [name_shape1, name_sum_axes], [name_dim1g])
            yield helper.make_node(
                'Gather', [name_shape2, name_sum_axes], [name_dim2g])
        else:
            name_dim1 = root + "_dim1"
            name_dim2 = root + "_dim2"
            name_dim1g = name_dim1 + 'g'
            name_dim2g = name_dim2 + 'g'
            concat_left.append(name_dim1)
            concat_right.append(name_dim2)
            yield helper.make_node(
                'Gather', [name_shape1, name_sum_axes], [name_dim1g])
            yield helper.make_node(
                'Gather', [name_shape2, name_sum_axes], [name_dim2g])
            yield helper.make_node(
                'ReduceProd', [name_dim1g], [name_dim1], keepdims=1)
            yield helper.make_node(
                'ReduceProd', [name_dim2g], [name_dim2], keepdims=1)

        # *shape1, *shape2
        name_agg_shape1 = root + "_resh1"
        name_agg_shape2 = root + "_resh2"
        yield helper.make_node(
            'Concat', concat_left, [name_agg_shape1], axis=0)
        yield helper.make_node(
            'Concat', concat_right, [name_agg_shape2], axis=0)

        # m1sh = m1.reshape((dim0, dimb, dim1))
        # m2sh = m2.reshape((dim0b, dimb, dim2))
        name_agg1 = root + "_aresh1"
        name_agg2 = root + "_aresh2"
        yield helper.make_node('Reshape', [name1, name_agg_shape1], [name_agg1])
        yield helper.make_node('Reshape', [name2, name_agg_shape2], [name_agg2])

        # dot = m1sh @ np.transpose(m2sh, (0, 2, 1))
        name_agg2_tr = root + "_aresh2_tr"
        yield helper.make_node(
            'Transpose', [name_agg2], [name_agg2_tr], perm=[0, 2, 1])

        name_dot = root + "_dot"
        yield helper.make_node(
            'MatMul', [name_agg1, name_agg2_tr], [name_dot])

        # new_shape = ([max(m1.shape[i], m2.shape[i]) for i in batch_axes] +
        #      [m1.shape[i] for i in left if i not in batch_axes] +
        #      [m2.shape[i] for i in right if i not in batch_axes])
        concat_final = []
        if len(batch_axes) > 0:
            name_max_dim = root + "_max_dim"
            concat_final.append(name_max_dim)
            yield helper.make_node(
                'Max', [name_dim0g, name_dim0bg], [name_max_dim])

        left_set = list(sorted(set(left) - (set(batch_axes) & set(left))))
        if len(left_set) > 0:
            name_left_dim = root + "_left_dim"
            name_left_set = root + "_left_set"
            yield numpy_helper.from_array(
                np.array(left_set, dtype=np.int64), name=name_left_set)
            yield helper.make_node(
                'Gather', [name_shape1, name_left_set], [name_left_dim])
            concat_final.append(name_left_dim)

        right_set = list(sorted(set(right) - (set(batch_axes) & set(right))))
        if len(right_set) > 0:
            name_right_dim = root + "_right_dim"
            name_right_set = root + "_right_set"
            yield numpy_helper.from_array(
                np.array(right_set, dtype=np.int64), name=name_right_set)
            yield helper.make_node(
                'Gather', [name_shape2, name_right_set], [name_right_dim])
            concat_final.append(name_right_dim)

        name_new_shape = root + '_new_shape'
        diff = (
            self.full_dim -
            (len(batch_axes) + len(left_set) + len(right_set)))
        if diff > 0:
            names_ones = root + "_ones"
            yield numpy_helper.from_array(
                np.array([1 for i in range(diff)], dtype=np.int64),
                name=names_ones)
            concat_final.append(names_ones)

        yield helper.make_node(
            'Concat', concat_final, [name_new_shape], axis=0)

        name_final = root + '_final'
        yield helper.make_node(
            'Reshape', [name_dot, name_new_shape], [name_final])

    def to_onnx(self, names, opset=None, **kwargs):
        """
        Converts this node into ONNX. Enumerates all ONNX node
        which participate to the conversion. The last one
        is the final output.

        :param names: dictionary where to find already converted name
        :param opset: opset
        :param kwargs: additional parameter for the conversion
        :return: output
        """
        if opset is None:
            opset = onnx_opset_version()
        method_name = "_to_onnx_%s" % self.name
        meth = getattr(self, method_name, None)
        if meth is None:
            if self.name.endswith("_mm"):
                raise NotImplementedError(
                    "to_onnx not implemented for %r."
                    "You should call method simplify_mm_nodes "
                    "to remove it." % self.name)
            raise NotImplementedError(
                "to_onnx not implemented for %r." % self.name)
        for node in meth(names, opset=opset, **kwargs):
            if hasattr(node, 'output'):
                names[id(self)] = node.output[0]
            yield node


class GraphEinsumSubOp:
    """
    Class gathering all nodes produced to explicit einsum
    operators.

    :param letters: list of distinct letters
    :param mat: matrix, see *analyse_einsum_equation*
    :param lengths: lengths of every input
    :param duplicates: see *analyse_einsum_equation*
    """

    def __init__(self, letters, mat, lengths, duplicates):
        self._nodes = {}
        self._mark = {}
        self._ops = []
        self._inputs = {}
        self.last_op = None
        self.last_added_op = None
        self.metadata = dict(
            letters=letters, mat=mat, lengths=lengths,
            mat0=mat.copy(), duplicates=duplicates)

    def append(self, op):
        """
        Adds one input or result.

        :param op: integer (an input) or an instance of class *EinsumSubOp*.
        :return: op or None if op is an integer
        """
        if isinstance(op, int):
            if op in self._nodes:
                raise RuntimeError("Key %d already added." % op)
            self._nodes[op] = op
            self.last_added_op = op
            self._inputs[op] = op
            return None
        if isinstance(op, EinsumSubOp):
            if op in self._nodes:
                raise RuntimeError(
                    "Key %d already added, op=%r." % (id(op), op))
            self._nodes[id(op)] = op
            self._ops.append(op)
            self.last_added_op = op
            return op
        raise TypeError("Unexpected type %r." % type(op))

    def mark_last_node(self):
        """
        Marks the last node as the final output.
        """
        if self.last_added_op is None:
            raise RuntimeError("last_added_op is None.")
        self.mark(-1, self.last_added_op)

    def mark(self, i, op):
        """
        Marks one input or result as an intermediate result
        after a full einsum step.

        :param op: integer (an input) or an instance of class *EinsumSubOp*.
        """
        if not isinstance(i, int):
            raise TypeError("i must an integer not %r." % type(i))
        if i != -1 and i not in self._inputs:
            raise RuntimeError(
                "Input %d was not registered in %r." % (i, self._inputs))
        if isinstance(op, EinsumSubOp):
            if id(op) not in self._nodes:
                raise RuntimeError(
                    "Key %d not found, op=%r." % (id(op), op))
            self._mark[i] = op
            self._mark[id(op)] = i
            self.last_op = op
        else:
            raise TypeError("Unexpected type %r." % type(i))

    def __iter__(self):
        "Iterates on nodes."
        for op in self._ops:
            yield op

    def to_dot(self, **kwargs):
        """
        Produces a graph in :epkg:`dot`.

        :param kwargs: additional graph option
        :return: string
        """
        options = {
            'orientation': 'portrait',
            'ranksep': '0.25',
            'nodesep': '0.05',
            'width': '0.5',
            'height': '0.1',
            'size': '5',
            'node': '[shape=record]',
        }
        options.update(kwargs)

        def d2s(d):
            it = []
            for k, v in sorted(d.items()):
                it.append("%s=%s" % (k, v))
            return " ".join(it)

        def d2sd(d):
            it = []
            for k, v in sorted(d.items()):
                if len(v) > 1:
                    it.append("%s=%s" % (k, ",".join(map(str, v))))
            return " ".join(it)

        rows = ["digraph{"]
        for k, v in options.items():
            if isinstance(v, str) and "[" in v:
                rows.append("{} {};".format(k, v))
            else:
                rows.append("{}={};".format(k, v))
        for k, v in self._nodes.items():
            if isinstance(v, int):
                let = [(r, self.metadata['letters'][i])
                       for i, r in enumerate(self.metadata['mat0'][v])
                       if r != -1]
                dup = self.metadata['duplicates'][v]
                if dup is None:
                    dup = ""
                else:
                    dup = " - %s" % d2sd(dup)
                let.sort()
                letters = "".join(_[1] for _ in let)
                lab = "input %d\\\\n%s\\\\n%s%s" % (
                    v, letters, str(self.metadata['mat0'][v]), dup)
                sk = v
            else:
                lab = "%s\\\\n%s" % (v.name, d2s(v.kwargs))
                sk = id(v)

            if sk in self._mark and isinstance(self._mark[sk], int):
                la = self._mark[sk]
                lab = lab.replace("\\\\n", " - I%d\\\\n" % la)
                s = ('%d [label="%s" style=filled fillcolor=red];' % (k, lab))
            else:
                s = '%d [label="%s"];' % (k, lab)
            rows.append(s)
            if not hasattr(v, 'inputs'):
                continue
            for i in v.inputs:
                vid = i if isinstance(i, int) else id(i)
                s = "%d -> %d;" % (vid, k)
                rows.append(s)
        rows.append("}")
        return "\n".join(rows)

    def clean_unused_nodes(self):
        """
        Cleans nodes with unused outputs.
        """

        def iteration(it):
            # Walks through all nodes.
            is_used = {}
            for node in self._ops:
                if not isinstance(node, EinsumSubOp):
                    continue
                if id(node) not in is_used:
                    is_used[id(node)] = []
                for inp in node.inputs:
                    if not isinstance(inp, EinsumSubOp):
                        continue
                    idn = id(inp)
                    if idn not in is_used:
                        is_used[idn] = []
                    is_used[idn].append(id(node))

            # Remove unused nodes.
            removed = []
            for k, v in is_used.items():
                if len(v) == 0:
                    removed.append(k)
            removed = set(removed)
            i_rem = []
            for i, op in enumerate(self._ops):
                if not isinstance(op, EinsumSubOp):
                    continue
                if id(op) in removed and id(op) not in self._mark:
                    i_rem.append((i, id(op)))
            for i, idn in reversed(i_rem):
                del self._ops[i]
                del self._nodes[idn]
            return len(i_rem) > 0

        it = 1
        while iteration(it):
            it += 1

        self.last_op = None
        self.last_added_op = None

    def simplify_mm_nodes(self):
        """
        Node name suffixed by `mm` are an artifact to keep
        the graph consistent while building it. They can
        now be replaced by the equivalent node without suffix `mm`.
        """
        for op in self:
            if not isinstance(op, EinsumSubOp):
                continue
            if op.name.endswith('_mm'):
                if len(op.inputs) != 2:
                    raise RuntimeError(
                        "Expecting 2 inputs for node %r not %r id=%r." % (
                            op.name, len(op.inputs), id(op)))
                op.name = op.name[:-3]
                op.inputs = op.inputs[:1]

    def to_onnx(self, output, *inputs, proto_type=None, opset=None, **kwargs):
        """
        Converts the graph into ONNX.

        :param output: output name
        :param inputs: input names
        :param proto_type: type used for all operators
        :param opset: desired opset, None for the last one
        :param kwargs: additional parameter to use when building
            the ONNX graph
        :return: ONNX graph
        """
        # inputs
        if opset is None:
            opset = onnx_opset_version()
        onx_inputs = []
        if proto_type is None:
            proto_type = TensorProto.FLOAT
        lengths = self.metadata['lengths']
        for inp, le in zip(inputs, lengths):
            onx_inputs.append(helper.make_tensor_value_info(
                inp, proto_type, [None for i in range(le)]))

        # output
        onx_output = helper.make_tensor_value_info(
            output, proto_type, [None for i in range(lengths[-1])])

        # nodes
        names = {i: name for i, name in enumerate(inputs)}
        nodes = []
        inits = []
        for op in self:
            for onx_node in op.to_onnx(names, opset=opset):
                if hasattr(onx_node, 'output'):
                    nodes.append(onx_node)
                else:
                    inits.append(onx_node)

        # last node
        last_node = nodes[-1]
        nodes.append(helper.make_node(
            'Identity', [last_node.output[0]], [output]))

        # Builds the graph
        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid('', opset)],
            ir_version=kwargs.get('ir_version', OPSET_TO_IR_VERSION.get(opset, 6)),
            producer_name=kwargs.get('producer_name', 'tensorflow-onnx'),
            producer_version=kwargs.get('producer_version', "0.0.dev"),
            graph=helper.make_graph(
                name=kwargs.get('name', 'einsum'),
                inputs=onx_inputs, outputs=[onx_output],
                initializer=inits, nodes=nodes))
        return model


def analyse_einsum_equation(equation):
    """
    Analyses an einsum equation.

    :param equation: :epkg:`numpy:einsum` equation
    :return: three results, list of letters,
        a matrix (see below), lengths of each components,
        duplicates

    The returned a matrix is defined as follows:

    .. math::

        m_{ij}=\\left\\{\\begin{array}{ll}-1 &
        \\text{if letter j is involved in input i} \\\\
        p & \\text{p is position of letter j in equation i}
        \\end{array}\\right.
    """
    spl = equation.strip(' ,').split("->")
    if len(spl) != 2 or len(spl[1]) == 0 or len(spl[0]) == 0:
        raise NotImplementedError(
            "The function only implements the case when there are "
            "two sides in the equation: %r." % equation)
    inputs = list(map(lambda s: s.strip(), spl[0].split(',')))
    output = spl[1]
    all_letters = set(inputs[0])

    # Set of letters
    for inp in inputs[1:]:
        all_letters |= set(inp)
    letters = list(sorted(all_letters))
    for c in letters:
        if not(('a' <= c <= 'z') or ('A' <= c <= 'Z')):
            raise ValueError(
                "Equation %r must only contain lower or upper letters "
                "but %r is not." % (equation, c))

    rev = {c: i for i, c in enumerate(letters)}
    for c in output:
        if c not in letters:
            raise ValueError(
                "Output contains one unexpected letter %r in "
                "equation %r." % (c, equation))
    mat = np.full((len(inputs) + 1, len(letters)), -1, dtype=np.int8)
    for i, inp in enumerate(inputs):
        for k, c in enumerate(inp):
            mat[i, rev[c]] = k
    for k, c in enumerate(output):
        mat[len(inputs), rev[c]] = k
    lengths = [len(inp) for inp in inputs]
    lengths.append(len(output))

    # Look for duplicates
    duplicates = []
    for inp in inputs + [output]:
        if len(inp) == len(set(inp)):
            duplicates.append(None)
            continue
        # There is some duplicates.
        counts = {}
        for i, c in enumerate(inp):
            if c in counts:
                counts[c].append(i)
            else:
                counts[c] = [i]
        duplicates.append(counts)

    return "".join(letters), mat, lengths, duplicates


def decompose_einsum_equation(equation, *shapes):
    """
    Decomposes an equation used in :epkg:`numpy:einsum` knowing
    the input shapes. It returns a sequence of operations
    to do to compute the results.

    :param equation: a string
    :param shapes: sequence of input shapes
    :return: instance of class *GraphEinsumSubOp*

    Available operations: *expand_dims*, *transpose*, *matmul*, *reduce_sum*,
    *id*, *squeeze*, *diagonal*. It analyses an equation and produces a graph
    where node are instance of class *EinsumSubOp*.
    """
    graph = _decompose_einsum_equation(
        equation, *shapes, op_matmul='batch_dot')

    # Last step: clean unused nodes.
    graph.mark_last_node()
    graph.simplify_mm_nodes()
    graph.clean_unused_nodes()
    return graph



def is_transpose_identity(perm):
    """
    Tells if the permutation *perm* does nothing (itentity).

    :param perm: permutation
    :return: boolean
    """
    return list(perm) == list(range(len(perm)))


def _basic_verification(lengths, shapes, equation):
    if len(lengths) - 1 != len(shapes):
        raise ValueError(
            "Equation %r has %d inputs but %d shapes are given."
            "" % (equation, len(lengths), len(shapes)))
    for i, (le, sh) in enumerate(zip(lengths, shapes)):
        if le != len(sh):
            raise ValueError(
                "Inputs %d has %d dimensions but shapes %r has %d "
                " in equation %r." % (i, le, sh, len(sh), equation))


def _apply_transpose_reshape(op, row):
    """
    Put all dimensions in the same order.

    :param op: integer (for one input) or an operator
    :param row: letter involved in this input (as a vector of binaries)
    :return: last created operator
    """
    axes = []
    p = 0
    perm = []
    for i, r in enumerate(row):
        if r == -1:
            axes.append((p, i))
        else:
            p += 1
            perm.append((r, i))
    op = EinsumSubOp(len(row), 'expand_dims', op, axes=tuple(axes))
    yield op
    perm.sort()
    p = 0
    new_perm = np.arange(len(row))
    for i, r in enumerate(row):
        if r == -1:
            continue
        new_perm[perm[p][1]] = i
        p += 1
    if not is_transpose_identity(new_perm):
        op = EinsumSubOp(len(row), 'transpose', op, perm=tuple(new_perm))
        yield op


def _apply_squeeze_transpose(op, row_last, row_output):
    """
    Puts output dimension in the expected order.
    """
    perm = []
    sq = []
    for i, d in enumerate(row_output):
        if d == -1:
            sq.append(i)
        else:
            perm.append((d, i))
    perm.sort()
    new_perm = np.arange(len(row_last))
    p = 0
    for i, d in enumerate(row_output):
        if d == -1:
            continue
        new_perm[i] = perm[p][1]
        p += 1
    perm = [p[1] for p in perm]
    if not is_transpose_identity(new_perm):
        op = EinsumSubOp(len(row_last), 'transpose', op,
                         perm=tuple(new_perm))
        yield op
    if len(sq) > 0:
        op = EinsumSubOp(len(row_last), 'squeeze', op, axes=tuple(sq))
        yield op


def _apply_einsum_matmul(fd, op1, op2, axes, left, right, ndim,
                         op_matmul, row1, row2):
    """
    Decomposes the generic matrix multiplication into numpy operations
    depending on the operator to use for matrix multiplication
    *op_matmul* (see *decompose_einsum_equation*).
    """
    allowed = {'matmul', 'batch_dot', 'dot'}
    if op_matmul not in allowed:
        raise ValueError(
            "Unknown operator op_matmul=%r not in %r." % (op_matmul, allowed))
    if op_matmul == 'matmul':
        yield EinsumSubOp(fd, 'matmul', op1, op2,
                          axes=axes, left=left, right=right, ndim=ndim)

    elif len(axes) == 0 and len(set(left) & set(right)) == 0:
        yield EinsumSubOp(fd, 'mul', op1, op2)

    elif (len(set(axes) & set(left)) == 0 and
            len(set(axes) & set(right)) == 0):

        # No intersection between axes and right: matrix multiplication
        all_axes = set(left) | set(right) | set(axes)
        common_axes = list(set(left) & set(right))
        for i in range(ndim):
            if i not in all_axes:
                common_axes.append(i)
        common_axes.sort()

        # ReduceSum*
        has_dim = set(i for i in range(len(row1)) if row1[i] >= 0)
        right_no_left = (set(right) & has_dim) - \
            (set(right) & (set(left) | set(axes)))
        if right_no_left:
            op1 = EinsumSubOp(fd, 'reduce_sum_mm', op1, op2,
                              axes=tuple(sorted(right_no_left)))
            yield op1

        has_dim = set(i for i in range(len(row2)) if row2[i] >= 0)
        left_no_right = (set(left) & has_dim) - \
            (set(left) & (set(right) | set(axes)))
        if left_no_right:
            op2 = EinsumSubOp(fd, 'reduce_sum', op2,
                              axes=tuple(sorted(left_no_right)))
            yield op2

        # Transpose
        i_axes = [(-1 if i in common_axes
                   else (1 if i in axes else 0), i)
                  for i in range(ndim)]
        i_axes.sort()
        perm = [_[1] for _ in i_axes]
        perm_left = [i for i in range(len(perm)) if perm[i] in left]
        perm_right = [i for i in range(len(perm)) if perm[i] in right]
        if not is_transpose_identity(perm):
            op1 = EinsumSubOp(fd, 'transpose_mm', op1, op2, perm=tuple(perm))
            yield op1
            op2 = EinsumSubOp(fd, 'transpose', op2, perm=tuple(perm))
            yield op2

        # Reshape
        all_axes = list(range(0, ndim))
        new_axes = all_axes[-len(axes):] if len(axes) > 0 else []
        new_common_axes = all_axes[:len(common_axes)]
        not_in_both = []
        for i in range(0, ndim):
            if i not in left and i not in right and i not in common_axes:
                not_in_both.append(i)

        op = EinsumSubOp(fd, 'batch_dot', op1, op2,
                         batch_axes=tuple(new_common_axes),
                         keep_axes=None, sum_axes=tuple(new_axes),
                         left=tuple(perm_left), right=tuple(perm_right),
                         ndim=ndim)
        yield op

        # Transpose again
        ordered_axes = (common_axes +
                        list(i for i in left if i not in right) +
                        list(i for i in right if i not in left) +
                        not_in_both)
        rev_perm = [(a, i) for i, a in enumerate(ordered_axes)]
        rev_perm.sort()
        rev_perm = [p[1] for p in rev_perm]

        if not is_transpose_identity(rev_perm):
            op_unused = EinsumSubOp(fd, 'transpose_mm', op1,
                                    op, perm=tuple(rev_perm))
            yield op_unused
            op = EinsumSubOp(fd, 'transpose', op, perm=tuple(rev_perm))
            yield op
    else:
        raise NotImplementedError(
            "axes and right or left have axes in common, "
            "axes=%r left=%r right=%r ndim=%r." % (
                axes, left, right, ndim))


def _decompose_einsum_equation(equation, *shapes, op_matmul='batch_dot'):
    """
    :param op_matmul: which operator to use for matrix multiplication,
        a single operator *matmul*, or *batch_dot* with *transposes*,
        *reduce_sum*, or just *dot*
    """
    letters, mat, lengths, duplicates = analyse_einsum_equation(equation)
    if len(letters) != mat.shape[1]:
        raise RuntimeError(  # pragma: no cover
            "Unexpected number of letters %r, shape=%r." % (
                letters, mat.shape))
    if len(shapes) > 0:
        _basic_verification(lengths, shapes, equation)
    else:
        shapes = [(2,) * le for le in lengths[:-1]]

    # last_row, current_row (row = shape)
    rows = np.full((2, mat.shape[1]), -1)
    graph = GraphEinsumSubOp(letters, mat, lengths, duplicates)
    fd = mat.shape[1]
    for i, sh in enumerate(shapes):
        graph.append(i)

        # Input matrix aligned to the same dimensions.
        op = EinsumSubOp(fd, 'id', i)
        op.compute_output_row(rows[1, :], mat[i, :])
        marked = graph.append(op)

        duplicate = duplicates[i]
        if duplicate is not None:
            # Diagonal
            diag = []
            for _, v in duplicate.items():
                if len(v) == 1:
                    continue
                diag.append((v[0], tuple(v)))
            op = EinsumSubOp(fd, 'diagonal', op, diag=diag)
            op.compute_output_row(rows[1, :], mat[i, :])
            tr_row = rows[1, :]
            marked = graph.append(op)
        else:
            diag = None
            tr_row = mat[i]

        for op in _apply_transpose_reshape(op, tr_row):
            op.compute_output_row(rows[1, :])
            marked = graph.append(op)

        # Reduction? (a dimension not used later)
        red = []
        for d in range(0, mat.shape[1]):
            if (mat[i + 1:, d].max() == -1 and rows[1, d] != -1 and
                    rows[0, d] == -1):
                red.append(d)
        if len(red) > 0:
            op = EinsumSubOp(fd, 'reduce_sum',
                             graph.last_added_op, axes=tuple(red))
            op.compute_output_row(rows[1, :])
            marked = graph.append(op)

        if graph.last_op is not None:
            # Matrix multiplication?
            common_dims = []
            left = []
            right = []
            for d in range(0, mat.shape[1]):
                if rows[:, d].min() >= 0:
                    if mat[i + 1:, d].max() >= 0:
                        left.append(d)
                        right.append(d)
                    else:
                        common_dims.append(d)
                else:
                    if rows[0, d] >= 0:
                        left.append(d)
                    if rows[1, d] >= 0:
                        right.append(d)
            for iop in _apply_einsum_matmul(
                    fd, graph.last_op, op, axes=tuple(common_dims),
                    left=tuple(left), right=tuple(right),
                    ndim=rows.shape[1], op_matmul=op_matmul,
                    row1=rows[0, :], row2=rows[1, :]):
                op = iop
                op.compute_output_row(rows[0, :], rows[1, :], ab=True)
                marked = graph.append(op)

        # End
        graph.mark(i, marked)
        rows[0, :] = rows[1, :]

    # Final output
    if mat[len(shapes), :].max() >= 0:
        rows[1, :] = mat[len(shapes), :]
        red = []
        for d in range(0, mat.shape[1]):
            if rows[0, d] > 0 and rows[1, d] == -1:
                red.append(d)
            elif rows[0, d] == -1 and rows[1, d] >= 0:
                raise RuntimeError(
                    "Issue in equation %r, variable %d, last_result is %r, "
                    "output is %r." % (equation, d, rows[0, :], rows[1, :]))
        if len(red) > 0:
            op = EinsumSubOp(fd, 'reduce_sum', op, axes=tuple(red))
            graph.append(op)
            op.compute_output_row(rows[1, :])

        # Removes empty axes.
        for op in _apply_squeeze_transpose(op, rows[1, :], mat[len(shapes), :]):
            op.compute_output_row(rows[1, :])
            graph.append(op)
    return graph
