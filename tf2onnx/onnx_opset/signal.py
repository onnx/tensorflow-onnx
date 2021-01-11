# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
signal
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from onnx import onnx_pb
from onnx.numpy_helper import to_array
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

def make_dft_constant(length, dtype, fft_length):
    n = np.arange(length)
    k = n.reshape((length, 1)).astype(np.float64)
    mat = np.exp(-2j * np.pi * k * n / length)
    mat = mat[:fft_length // 2 + 1]
    both = np.empty((2,) + mat.shape, dtype=dtype)
    both[0, :, :] = np.real(mat)
    both[1, :, :] = np.imag(mat)
    return both


@tf_op("RFFT")
class RFFTOp:
    # support more dtype

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        """
        Inspired from `Python implementation of RFFT
        <https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/>`_.

        Complex version:

        ::

            import numpy as np

            def _DFT_cst(N, fft_length):
                n = np.arange(N)
                k = n.reshape((N, 1)).astype(np.float64)
                M = np.exp(-2j * np.pi * k * n / N)
                return M[:fft_length // 2 + 1]

            def DFT(x, fft_length=None):
                if len(x.shape) == 1:
                    x = x.reshape((-1, 1))
                else:
                    x = x.T
                if fft_length is None:
                    fft_length = x.shape[0]
                cst = _DFT_cst(x.shape[0], fft_length)
                return np.dot(cst, x).T

        Real version, first axis is (real, imag) part:

        ::

            import numpy as np

            def _DFT_real_cst(N, fft_length):
                n = np.arange(N)
                k = n.reshape((N, 1)).astype(np.float64)
                M = np.exp(-2j * np.pi * k * n / N)
                M = M[:fft_length // 2 + 1]
                both = np.empty((2,) + M.shape)
                both[0, :, :] = np.real(M)
                both[1, :, :] = np.imag(M)
                return both

            def DFT_real(x, fft_length=None):
                if len(x.shape) == 1:
                    x = x.reshape((-1, 1))
                else:
                    x = x.T
                if fft_length is None:
                    fft_length = x.shape[0]
                cst = _DFT_real_cst(x.shape[0], fft_length)
                res = np.dot(cst, x)
                return np.transpose(res, (0, 2, 1))
        """
        supported_dtypes = [
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.DOUBLE,
            onnx_pb.TensorProto.COMPLEX64,
            onnx_pb.TensorProto.COMPLEX128,
        ]
        consumers = ctx.find_output_consumers(node.output[0])
        consumer_types = set(op.type for op in consumers)
        utils.make_sure(
            consumer_types == {'ComplexAbs'},
            "Current implementation of RFFT only allows ComplexAbs as consumer not %r",
            consumer_types)

        onnx_dtype = ctx.get_dtype(node.input[0])
        utils.make_sure(onnx_dtype in supported_dtypes, "Unsupported input type.")
        shape = ctx.get_shape(node.input[0])
        np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)
        shape_n = shape[-1]
        utils.make_sure(len(node.input) == 2, "Two inputs expected not %r", len(node.input))

        # This input should be a constant.
        fft_length_name = node.input[1]
        node_fft_length = ctx.get_node_by_output(fft_length_name, search_in_parent_graphs=True)
        utils.make_sure(node_fft_length.type == 'Const',
                        "fft_length should be a constant, the other case is not implemented yet.")
        value = node_fft_length.get_attr("value")
        value_array = to_array(value.t)
        utils.make_sure(value_array.shape == (1,), "Unexpected shape for fft_length (%r)", value_array.shape)
        fft_length = value_array[0]

        # TODO: handle this parameter when onnx.helper.make_node is fixed.
        # Tcomplex = node.get_attr("Tcomplex")

        if np_dtype == np.float16:
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float16)
            np_dtype = np.float16
        elif np_dtype in (np.float32, np.complex64):
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float32)
            np_dtype = np.float32
        else:
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float64)
            np_dtype = np.float64

        real_imag_part = make_dft_constant(shape_n, np_dtype, fft_length)
        onx_real_imag_part = ctx.make_const(
            name=utils.make_name('cst_rfft_%d' % shape_n), np_val=real_imag_part)

        shapei = list(np.arange(len(shape)))
        perm = shapei[:-2] + [shapei[-1], shapei[-2]]
        trx = ctx.make_node(
            "Transpose", inputs=[node.input[0]], attr=dict(perm=perm),
            name=utils.make_name(node.name + 'tr'))

        ctx.remove_node(node.name)
        mult = ctx.make_node(
            "MatMul", inputs=[onx_real_imag_part.name, trx.output[0]],
            name=utils.make_name('CPLX_' + node.name + 'rfft'))

        new_shape = [2] + list(shape)
        shapei = list(np.arange(len(new_shape)))
        perm = shapei[:-2] + [shapei[-1], shapei[-2]]
        last_node = ctx.make_node(
            "Transpose", inputs=[mult.output[0]], attr=dict(perm=perm),
            name=utils.make_name('CPLX_' + node.name + 'rfft'),
            shapes=[new_shape], dtypes=[res_onnx_dtype])

        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()


@tf_op("ComplexAbs")
class ComplexAbsOp:
    # support more dtype

    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        """
        Computes the modules of a complex.
        If the matrix dtype is not complex64 or complex128,
        it assumes the first dimension means real part (0)
        and imaginary part (1, :, :...).
        """
        supported_dtypes = [
            onnx_pb.TensorProto.FLOAT,
            onnx_pb.TensorProto.FLOAT16,
            onnx_pb.TensorProto.DOUBLE,
            onnx_pb.TensorProto.COMPLEX64,
            onnx_pb.TensorProto.COMPLEX128,
        ]
        onnx_dtype = ctx.get_dtype(node.input[0])
        utils.make_sure(onnx_dtype in supported_dtypes, "Unsupported input type.")
        shape = ctx.get_shape(node.input[0])
        np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)
        utils.make_sure(shape[0] == 2, "ComplexAbs expected the first dimension to be 2 but shape is %r", shape)

        ind0 = ctx.make_const(name=utils.make_name('cst0'), np_val=np.array([0], dtype=np.int64))
        ind1 = ctx.make_const(name=utils.make_name('cst1'), np_val=np.array([1], dtype=np.int64))
        p2 = ctx.make_const(name=utils.make_name('p2'), np_val=np.array([2], dtype=np_dtype))

        real_part = ctx.make_node(
            'Gather', inputs=[node.input[0], ind0.name], attr=dict(axis=0),
            name=utils.make_name('Real_' + node.name))
        imag_part = ctx.make_node(
            'Gather', inputs=[node.input[0], ind1.name], attr=dict(axis=0),
            name=utils.make_name('Imag_' + node.name))

        real_part2 = ctx.make_node(
            'Pow', inputs=[real_part.output[0], p2.name],
            name=utils.make_name(real_part.name + 'p2p'))

        imag_part2 = ctx.make_node(
            'Pow', inputs=[imag_part.output[0], p2.name],
            name=utils.make_name(imag_part.name + 'p2p'))

        ctx.remove_node(node.name)
        add = ctx.make_node(
            "Add", inputs=[real_part2.output[0], imag_part2.output[0]],
            name=utils.make_name('ComplexAbs_' + node.name))

        squeezed = GraphBuilder(ctx).make_squeeze(
            {'data': add.output[0], 'axes': [0]}, name=utils.make_name('ComplexAbs' + node.name), return_node=True)

        last_node = ctx.make_node(
            "Sqrt", inputs=squeezed.output[:1],
            name=utils.make_name('ComplexAbs' + node.name),
            shapes=[shape[1:]], dtypes=[onnx_dtype])

        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        cls.any_version(1, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        cls.any_version(13, ctx, node, **kwargs)
