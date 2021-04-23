# SPDX-License-Identifier: Apache-2.0

"""Rewrites operator einsum into simple ONNX operators.
"""

from __future__ import unicode_literals
import numpy as np


def _numpy_extended_dot_equation(m1_dim, m2_dim, axes, left, right):
    """
    Returns the equation equivalent to an extended version
    of a matrix multiplication (see function *numpy_extended_dot*).

    :param m1: number of dimensions of the first matrix
    :param m2: number of dimensions of the second matrix
    :param axes: summation axes
    :param axes: summation axes
    :param left: left axes
    :param right: right axes
    :return: equation
    """
    if m1_dim != m2_dim:
        raise RuntimeError(
            "Matrices m1 and m2 must have the same number of dimensions, "
            "m1=%r, m2=%r." % (m1_dim, m2_dim))
    total = set(axes) | set(left) | set(right)
    if len(total) > m1_dim:
        raise ValueError(
            "Whole set of involved axes should be inferior to the number "
            "of dimensions: %r = {%r} | {%r} | {%r} has more than %d elements"
            "." % (total, axes, left, right, m1_dim))

    def _check_(axs, n):
        for a in axs:
            if a < 0 or a >= n:
                raise ValueError(
                    "One axis %d (in %r) is negative or above the maximum "
                    "dimension %d." % (a, axs, n))
    _check_(axes, m1_dim)
    _check_(left, m1_dim)
    _check_(right, m1_dim)

    l1 = [chr(i + 97) for i in range(m1_dim)]
    l2 = [chr(i + 97) for i in range(m1_dim)]
    l3 = [chr(i + 97) for i in range(m1_dim)]
    for a in left:
        l1[a] = l1[a].upper()
        l3[a] = l3[a].upper()
    for a in right:
        l2[a] = l2[a].upper()
        l3[a] = l3[a].upper()
    for a in axes:
        l1[a] = l1[a].lower()
        l2[a] = l2[a].lower()
        if a not in right:
            l3[a] = None
        else:
            l3[a] = l3[a].lower()
    eq = "%s,%s->%s" % ("".join(l1), "".join(l2),
                        "".join(s for s in l3 if s))
    return eq


def numpy_extended_dot(m1, m2, axes, left, right):
    """
    Extended version of a matrix multiplication (:epkg:`numpy:dot`)
    with two matrices *m1*, *m2* of the same dimensions.
    Loops over *left* axes for *m1* and *right* axes for *m2*,
    summation is done over *axes*.
    Other axes must be empty.
    This multiplication combines matrix multiplication (dot)
    and broadcasted multiplication term by term.

    :param m1: first matrix
    :param m2: second matrix
    :param axes: summation axes
    :param left: left axes
    :param right: right axes
    :return: output

    The dot product is equivalent to:

    ::

        m1 = np.arange(4).reshape((2, 2))
        m2 = m1 + 10
        print("dot product")
        print(m1 @ m2)

        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[1], left=[0], right=[2])
        print("extended dot product")
        print(dot)

    Empty axes should be squeezed to get identical results.
    Dot product when the second matrix is transposed.

    ::

        m1 = np.arange(4).reshape((2, 2))
        m2 = m1 + 10
        print("dot product")
        print(m1 @ m2.T)

        dm1 = m1.reshape((2, 1, 2))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1])
        print("extended dot product")
        print(dot)

    An example when right axes include the summation axis.

    ::

        m1 = np.arange(4).reshape((2, 2))
        m2 = m1 + 10
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1, 2])
        print(dot)

    Example in higher dimension:

    ::

        m1 = np.arange(8).reshape((2, 2, 2))
        m2 = m1 + 10

        dot = numpy_extended_dot(m1, m2, [1], [0], [2]))
        print(dot)

    The current implementation still uses :epkg:`numpy:einsum`
    but this should be replaced.
    """
    if m1.dtype != m2.dtype:
        raise TypeError(
            "Both matrices should share the same dtype %r != %r."
            "" % (m1.dtype, m2.dtype))
    eq = _numpy_extended_dot_equation(
        len(m1.shape), len(m2.shape), axes, left, right)
    output = np.einsum(eq, m1, m2)
    new_shape = list(output.shape)
    for a in axes:
        if a not in right:
            new_shape.insert(a, 1)
    return output.reshape(tuple(new_shape))


def numpy_diagonal(m, axis, axes):
    """
    Extracts diagonal coefficients from an array.

    :param m: input array
    :param axis: kept axis among the diagonal ones
    :param axes: diagonal axes (axis must be one of them)
    :return: output

    ::

        mat = np.arange(8).reshape((2, 2, 2))
        print(mat)
        diag = numpy_diagonal(mat, 1, [1, 2])
        print(diag)
    """
    if axis not in axes:
        raise RuntimeError(
            "axis %r must be in axes %r." % (axis, axes))
    shape = []
    new_shape = []
    for i, s in enumerate(m.shape):
        if i in axes:
            if i == axis:
                shape.append(s)
                new_shape.append(s)
            else:
                shape.append(1)
        else:
            shape.append(s)
            new_shape.append(s)

    # Extracts coefficients.
    output = np.empty(tuple(shape), dtype=m.dtype)
    index_in = [slice(s) for s in m.shape]
    index_out = [slice(s) for s in m.shape]
    for i in range(0, shape[axis]):
        for a in axes:
            index_in[a] = i
            index_out[a] = i if a == axis else 0
        output[tuple(index_out)] = m[tuple(index_in)]

    # Removes axis.
    return output.reshape(tuple(new_shape))


class EinsumSubOp:
    """
    Defines a sub operation used in Einsum decomposition.

    :param name: name (reshape, transpose, reduce_sum, matmul, id)
    :param inputs: inputs
    :param kwargs: arguments
    """
    _allowed = {'expand_dims', 'transpose', 'reduce_sum', 'matmul', 'id',
                'squeeze', 'diagonal'}

    def __init__(self, full_dim, name, *inputs, **kwargs):
        self.full_dim = full_dim
        self.name = name
        self.inputs = inputs
        self.kwargs = kwargs
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

    def __repr__(self):
        inps = ", ".join(map(str, self.inputs))
        kw = ", ".join("%s=%r" % (k, w) for k, w in self.kwargs.items())
        m = "%s(%r, %s, %s)" % (
            self.__class__.__name__, self.name, inps, kw)
        return m

    def dot_label(self):
        """
        Displays some informations useful to understand the operator.
        """
        if self.name == "matmul":
            ndim = self.kwargs['ndim']
            axes = self.kwargs['axes']
            left = self.kwargs['left']
            right = self.kwargs['right']
            eq = _numpy_extended_dot_equation(ndim, ndim, axes, left, right)
            eq = eq.replace(">", "\\\\>")
            return "~" + eq

    def _check_arg_(self, name, typ):
        if name not in self.kwargs:
            raise RuntimeError(
                "Parameter %r not found for operator %r." % (name, self.name))
        if not isinstance(self.kwargs[name], typ):
            raise TypeError(
                "Unexpected type %r for parameter %r and parameter %r."
                "" % (type(self.kwargs[name]), name, self.name))

    def _compute_output_row_id(self, row, row2=None):
        row[:] = row2[:]

    def _compute_output_row_transpose(self, row, row2=None):
        self._check_arg_('perm', tuple)
        if len(self.kwargs['perm']) != len(row):
            raise RuntimeError(
                "Unexpected permutation %r (row=%r)."
                "" % (self.kwargs['perm'], row))
        cpy = row.copy()
        for i, p in enumerate(self.kwargs['perm']):
            row[i] = cpy[p]

    def _compute_output_row_expand_dims(self, row, row2=None):
        self._check_arg_('axis', tuple)
        if row[self.kwargs['axis'][1]] != -1:
            raise RuntimeError(
                "Dimension should be -1 in row %r axis=%r." % (
                    row, self.kwargs['axis']))

    def _compute_output_row_reduce_sum(self, row, row2=None):
        self._check_arg_('axes', tuple)
        for a in self.kwargs['axes']:
            row[a] = -1

    def _compute_output_row_matmul(self, row, row2=None):
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

    def _compute_output_row_squeeze(self, row, row2=None):
        self._check_arg_('axes', tuple)
        for a in self.kwargs['axes']:
            row[a] = -1

    def _compute_output_row_diagonal(self, row, row2=None):
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

    def compute_output_row(self, row, row2=None):
        """
        Updates *row* based on the operator.
        """
        method_name = "_compute_output_row_%s" % self.name
        meth = getattr(self, method_name, None)
        if meth is None:
            raise NotImplementedError(
                "compute_output_row not implemented for %r." % self.name)
        meth(row, row2=row2)

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

    def _apply_id(self, data):
        self._check_inputs_(1)
        inp = self.inputs[0]
        output = self._get_data(data, inp)
        return output

    def _apply_diagonal(self, data):
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        diag = self.kwargs['diag']
        if len(diag) != 1:
            raise NotImplementedError(
                "Not implemented with more than one duplicated indice "
                "%r." % diag)
        diag0 = diag[0]
        output = numpy_diagonal(m, axis=diag0[0], axes=diag0[1])
        return output

    def _apply_expand_dims(self, data):
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        output = np.expand_dims(m, self.kwargs['axis'][0])
        return output

    def _apply_transpose(self, data):
        self._check_inputs_(1, True)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        output = np.transpose(m, self.kwargs['perm'])
        self._check_shape_(output)
        return output

    def _apply_matmul(self, data):
        self._check_inputs_(2)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        self._check_shape_(m1)
        self._check_shape_(m2)
        axes = self.kwargs['axes']
        left = self.kwargs['left']
        right = self.kwargs['right']
        output = numpy_extended_dot(m1, m2, axes, left, right)
        self._check_shape_(output)
        return output

    def _apply_reduce_sum(self, data):
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        axes = self.kwargs['axes']
        output = np.sum(m, axis=axes, keepdims=True)
        self._check_shape_(output)
        return output

    def _apply_squeeze(self, data):
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        axes = self.kwargs['axes']
        output = m
        for a in axes[::-1]:
            output = np.squeeze(output, axis=a)
        return output

    def apply(self, data):
        """
        Applies one operator on the data.

        :param data: dictionary storing the results
        """
        method_name = "_apply_%s" % self.name
        meth = getattr(self, method_name, None)
        if meth is None:
            raise NotImplementedError(
                "apply not implemented for %r." % self.name)
        output = meth(data)

        data[id(self)] = output
        return output


class GraphEinsumSubOp:
    """
    Class gathering all nodes produced to explicit einsum
    operators.
    """

    def __init__(self, letters, mat, lengths, duplicates):
        self._nodes = {}
        self._mark = {}
        self._ops = []
        self.last_op = None
        self.last_added_op = None
        self.metadata = dict(
            letters=letters, mat=mat, lengths=lengths,
            mat0=mat.copy(), duplicates=duplicates)

    def append(self, op):
        """
        Adds one input or result.

        :param op: integer (an input) or an instance of *EinsumSubOp*.
        :return: op or None if op is an integer
        """
        if isinstance(op, int):
            if op in self._nodes:
                raise RuntimeError("Key %d already added." % op)
            self._nodes[op] = op
            self.last_added_op = op
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

    def mark(self, i, op):
        """
        Marks one input or result as an intermediate result
        after a full einsum step.

        :param op: integer (an input) or an instance of *EinsumSubOp*.
        """
        if not isinstance(i, int):
            raise TypeError("i must an integer not %r." % type(i))
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
                extended_lab = ""
            else:
                lab = "%s\\\\n%s" % (v.name, d2s(v.kwargs))
                sk = id(v)
                extended_lab = v.dot_label()
                if extended_lab:
                    extended_lab = "\\\\n" + extended_lab

            if sk in self._mark and isinstance(self._mark[sk], int):
                la = self._mark[sk]
                lab = lab.replace("\\\\n", " - I%d\\\\n" % la)
                s = ('%d [label="%s%s" style=filled '
                     'fillcolor=red];' % (k, lab, extended_lab))
            else:
                s = '%d [label="%s%s"];' % (k, lab, extended_lab)
            rows.append(s)
            if not hasattr(v, 'inputs'):
                continue
            for i in v.inputs:
                vid = i if isinstance(i, int) else id(i)
                s = "%d -> %d;" % (vid, k)
                rows.append(s)
        rows.append("}")
        return "\n".join(rows)

    def apply_sequence(self, *inputs):
        """
        Applies a sequence of operations on a list of inputs.

        :param inputs: inputs:
        :return: output
        """
        data = {i: inp for i, inp in enumerate(inputs)}
        last = None
        for op in self:
            last = op.apply(data)
        if last is None:
            raise RuntimeError(
                "Sequence of operations is empty.")
        return last


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


def decompose_einsum_equation(equation, *shapes, strategy="simple"):
    """
    Decomposes an equation used in :epkg:`numpy:einsum` knowing
    the input shapes. It returns a sequence of operations
    to do to compute the results.

    :param equation: a string
    :param shapes: sequence of input shapes
    :param strategy: there are different way to decompose the equation,
        this parameters defines the way to do it (see below)
    :return: instance of *GraphEinsumSubOp*

    About *strategy*:
    * `'simple'`: align all dimensions in the alphabetical order

    Available operations: *expand_dims*, *transpose*, *matmul*, *reduce_sum*,
    *id*, *squeeze*, *diagonal*. It analyses an equation and produces a graph
    where node are instance of class *EinsumSubOp*.

    ::

        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        for op in seq:
            print(op)

    It can be better displayed as follows.
    
    ::

        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        print(seq.to_dot())
    """
    if len(shapes) == 0:
        raise ValueError("No input shapes.")
    for sh in shapes:
        if not isinstance(sh, tuple):
            raise TypeError(
                "All shapes must be tuples for %r is not." % sh)
    if strategy == "simple":
        return _decompose_einsum_equation_simple(equation, *shapes)
    raise ValueError("Unknown strategy %r." % strategy)


def apply_einsum_sequence(seq, *inputs):
    """
    Applies a sequence of operations on a list of inputs.
    The sequence of operations is produced by function
    *decompose_einsum_equation*.

    :param seq: sequence of operations
    :param inputs: inputs:
    :return: output

    ::

        m1 = np.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = np.arange(4).reshape((2, 2)) + 100
        m3 = np.arange(2 * 2).reshape((2, 2)) + 1000

        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        res = apply_einsum_sequence(seq, m1, m2)
        print(res)

    See notebook :ref:`einsumdecompositionrst`.
    """
    return seq.apply_sequence(*inputs)


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
    for a in reversed(axes):
        op = EinsumSubOp(len(row), 'expand_dims', op, axis=a)
        yield op
    perm.sort()
    p = 0
    new_perm = np.arange(len(row))
    for i, r in enumerate(row):
        if r == -1:
            continue
        new_perm[perm[p][1]] = i
        p += 1
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
    op = EinsumSubOp(len(row_last), 'transpose', op, perm=tuple(new_perm))
    yield op
    if len(sq) > 0:
        op = EinsumSubOp(len(row_last), 'squeeze', op, axes=tuple(sq))
        yield op


def _decompose_einsum_equation_simple(equation, *shapes):
    """
    Applies strategy simple of function *decompose_einsum_equation*.
    """
    letters, mat, lengths, duplicates = analyse_einsum_equation(equation)
    if len(letters) != mat.shape[1]:
        raise RuntimeError(  # pragma: no cover
            "Unexpected number of letters %r, shape=%r." % (
                letters, mat.shape))
    _basic_verification(lengths, shapes, equation)

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
                    common_dims.append(d)
                    if mat[i + 1:, d].max() >= 0:
                        left.append(d)
                        right.append(d)
                else:
                    if rows[0, d] >= 0:
                        left.append(d)
                    if rows[1, d] >= 0:
                        right.append(d)
            op = EinsumSubOp(fd, 'matmul', graph.last_op, op,
                             axes=tuple(common_dims),
                             left=tuple(left), right=tuple(right),
                             ndim=rows.shape[1])
            op.compute_output_row(rows[0, :], rows[1, :])
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
