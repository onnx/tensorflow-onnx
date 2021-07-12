# SPDX-License-Identifier: Apache-2.0


"""
signal
"""

import logging

import numpy as np
from onnx import onnx_pb, helper
from onnx.numpy_helper import to_array
from tf2onnx import utils
from tf2onnx.handler import tf_op
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

def make_dft_constant(length, dtype, fft_length):
    n = np.arange(length)
    k = n.reshape((length, 1)).astype(np.float64)
    utils.make_sure(fft_length > 0, "fft_length must be strictly positive but is %r.", fft_length)
    mat = np.exp(-2j * np.pi * k * n / fft_length)
    both = np.empty((2,) + mat.shape, dtype=dtype)
    both[0, :, :] = np.real(mat)
    both[1, :, :] = np.imag(mat)
    return both


class CommonFFTOp:
    supported_dtypes = [
        onnx_pb.TensorProto.FLOAT,
        onnx_pb.TensorProto.FLOAT16,
        onnx_pb.TensorProto.DOUBLE,
        onnx_pb.TensorProto.COMPLEX64,
        onnx_pb.TensorProto.COMPLEX128,
    ]

    @classmethod
    def any_version(cls, const_length, opset, ctx, node, axis=None,
                    fft_length=None, dim=None, onnx_dtype=None, shape=None,
                    input_name=None, **kwargs):
        """
        Inspired from `Python implementation of RFFT
        <https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/>`_.

        Complex version:

        ::

            import numpy as np

            def _DFT_cst(N, fft_length):
                n = np.arange(N)
                k = n.reshape((N, 1)).astype(np.float64)
                M = np.exp(-2j * np.pi * k * n / fft_length)
                return M

            def DFT(x, fft_length=None):
                if len(x.shape) == 1:
                    x = x.reshape((-1, 1))
                else:
                    x = x.T
                if fft_length is None:
                    fft_length = x.shape[0]
                cst = _DFT_cst(x.shape[0], fft_length)
                size = fft_length // 2 + 1
                return np.dot(cst[:, :fft_length], x[:fft_length]).T[:, :size]

        Real version, first axis is (real, imag) part:

        ::

            def _DFT_real_cst(N, fft_length):
                n = np.arange(N)
                k = n.reshape((N, 1)).astype(np.float64)
                M = np.exp(-2j * np.pi * k * n / fft_length)
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
                size = fft_length // 2 + 1
                cst = _DFT_real_cst(x.shape[0], fft_length)
                res = np.dot(cst[:, :, :fft_length], x[:fft_length])[:, :size, :]
                return np.transpose(res, (0, 2, 1))
        """
        if input_name is None:
            input_name = node.input[0]
            node_name = node.name
        else:
            node_name = input_name.split(':')[0]
        if axis is None:
            consumers = ctx.find_output_consumers(node.output[0])
            consumer_types = set(op.type for op in consumers)
            utils.make_sure(
                axis == 0 or consumer_types == {'ComplexAbs'},
                "Current implementation of RFFT or FFT only allows ComplexAbs as consumer not %r",
                consumer_types)

            onnx_dtype = ctx.get_dtype(input_name)
            utils.make_sure(onnx_dtype in CommonFFTOp.supported_dtypes, "Unsupported input type.")
            shape = ctx.get_shape(node.input[0])
            shape_n = shape[-1] if dim is None else dim
        else:
            shape_n = dim
            utils.make_sure(shape is not None, "shape must be known.")

        if onnx_dtype in (onnx_pb.TensorProto.COMPLEX64, onnx_pb.TensorProto.COMPLEX128):
            parent = ctx.get_node_by_output_in_current_graph(input_name)
            utils.make_sure(
                parent.type == 'Cast' and parent.get_attr_value('to') == onnx_dtype,
                "Current implementation of FFT or RFFT assumes the input is real or complex produced "
                "by a node Cast just before this one.")
            input_name = parent.input[0]
            onnx_dtype = ctx.get_dtype(input_name)

        np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)

        if np_dtype == np.float16:
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float16)
            np_dtype = np.float16
        elif np_dtype in (np.float32, np.complex64):
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float32)
            np_dtype = np.float32
        else:
            res_onnx_dtype = utils.map_numpy_to_onnx_dtype(np.float64)
            np_dtype = np.float64
            
        if const_length:
            # RFFT: length of FFT is known, some computation
            # (see function make_dft_constant)
            # can be done at conversion time and stored as constant
            utils.make_sure(axis is not None or len(node.input) == 2,
                            "Two inputs expected not %r", len(node.input) if axis is None else '?')

            # This input should be a constant.
            if fft_length is None:
                fft_length_name = node.input[1]
                node_fft_length = ctx.get_node_by_output(fft_length_name, search_in_parent_graphs=True)
                utils.make_sure(node_fft_length.type == 'Const',
                                "fft_length should be a constant, the other case is not implemented yet.")
                value = node_fft_length.get_attr("value")
                value_array = to_array(value.t)
                if axis is None:
                    utils.make_sure(
                        value_array.shape == (1,), "Unexpected shape for fft_length (%r)", value_array.shape)
                    fft_length = value_array[0]
                else:
                    utils.make_sure(
                        axis < len(value_array), "Inconsistent axis %r incompatible with fft_length=%r",
                        axis, value_array)
                    fft_length = value_array[axis]
                utils.make_sure(shape is None or fft_length <= shape[1], "Case fft_length > shape[1] is not implemented.")

            if axis is not None or fft_length < shape_n:
                real_imag_part = make_dft_constant(shape_n, np_dtype, fft_length)[:, :, :fft_length]
                
                if opset >= 10:
                    cst_axis = ctx.make_const(
                        name=utils.make_name('CPLX_csta'), np_val=np.array([-1], dtype=np.int64))
                    cst_zero = ctx.make_const(
                        name=utils.make_name('CPLX_cstz'), np_val=np.array([0], dtype=np.int64))
                    cst_length = ctx.make_const(
                        name=utils.make_name('CPLX_cstl'), np_val=np.array([fft_length], dtype=np.int64))
                    sliced_input = ctx.make_node(
                        "Slice", inputs=[input_name, cst_zero.name, cst_length.name, cst_axis.name],
                        name=utils.make_name('CPLX_S_' + node_name + 'rfft'))
                else:
                    sliced_input = ctx.make_node(
                        "Slice", inputs=[mult.output[0]], attr=dict(starts=[0], ends=[fft_length], axes=[-1]),
                        name=utils.make_name('CPLX_S_' + node_name + 'rfft'))
                input_name = sliced_input.output[0]
            else:
                size = fft_length // 2 + 1
                real_imag_part = make_dft_constant(shape_n, np_dtype, fft_length)[:, :size, :fft_length]

            onx_real_imag_part = ctx.make_const(
                name=utils.make_name('cst_rfft_%d' % shape_n), np_val=real_imag_part)
            onx_real_imag_part_name = onx_real_imag_part.name
        else:
            # FFT: length of FFT is unknown at conversion time, the matrix
            # created by function make_dft_constant must be
            # done in ONNX.
            utils.make_sure(axis is None, "Dynamic version of FFT is not implemented when axis != None.")
            dyn_shape_all = ctx.make_node("Shape", inputs=[input_name],
                                          name=utils.make_name('CPLX_' + node_name + 'shape'))
            m1_cst = ctx.make_const(name=utils.make_name('CPLX_m1'), np_val=np.array([-1], dtype=np.int64))
            dyn_shape = ctx.make_node('Gather', inputs=[dyn_shape_all.output[0], m1_cst.name])
            one_tensor = helper.make_tensor("value", res_onnx_dtype, dims=[1], vals=[1])
            cst_1 = ctx.make_node("ConstantOfShape", inputs=[dyn_shape.output[0]], attr={"value": one_tensor})
            just_0 = ctx.make_const(name=utils.make_name('CPLX1'), np_val=np.array([0], dtype=np.int64))
            rng1 = ctx.make_node("CumSum", inputs=[cst_1.output[0], just_0.name],
                                 name=utils.make_name('CPLX_' + node_name + 'range'))
            p1_cst = ctx.make_const(name=utils.make_name('CPLX_p1'), np_val=np.array([1], dtype=np_dtype))
            rng = ctx.make_node("Sub", inputs=[rng1.output[0], p1_cst.name],
                                name=utils.make_name('CPLX_' + node_name + 'range'))
            resh_cst = ctx.make_const(name=utils.make_name('CPLX_reshape'), np_val=np.array([1, -1], dtype=np.int64))
            rng_tr1 = ctx.make_node("Reshape", inputs=[rng.output[0], resh_cst.name],
                                    name=utils.make_name('CPLX_' + node_name + 'range'))
            resh_cst = ctx.make_const(name=utils.make_name('CPLX_reshape'), np_val=np.array([-1, 1], dtype=np.int64))
            rng_tr2 = ctx.make_node("Reshape", inputs=[rng.output[0], resh_cst.name],
                                    name=utils.make_name('CPLX_' + node_name + 'range'))
            rng_mat = ctx.make_node('MatMul', inputs=[rng_tr2.output[0], rng_tr1.output[0]],
                                    name=utils.make_name('CPLX_' + node_name + 'range2'))
            pi_cst = ctx.make_const(name=utils.make_name('CPLX_pi'), np_val=np.array([np.pi * 2], dtype=np_dtype))
            angle_pi = ctx.make_node("Mul", inputs=[rng_mat.output[0], pi_cst.name],
                                     name=utils.make_name('CPLX_' + node_name + 'angle_pi'))
            shape_cast = ctx.make_node('Cast', inputs=[dyn_shape.output[0]], attr={'to': res_onnx_dtype})
            angle_pibn = ctx.make_node("Div", inputs=[angle_pi.output[0], shape_cast.output[0]],
                                       name=utils.make_name('CPLX_' + node_name + 'angle'))
            if opset >= 13:
                angle = ctx.make_node("Unsqueeze", inputs=[angle_pibn.output[0], just_0.name],
                                      name=utils.make_name('CPLX_' + node_name + 'angles'))
            else:
                angle = ctx.make_node("Unsqueeze", inputs=[angle_pibn.output[0]],
                                      name=utils.make_name('CPLX_' + node_name + 'angles'),
                                      attr={'axes': [0]})
            rng_cos = ctx.make_node("Cos", inputs=[angle.output[0]],
                                    name=utils.make_name('CPLX_' + node_name + 'cos'))
            rng_sin = ctx.make_node("Sin", inputs=[angle.output[0]],
                                    name=utils.make_name('CPLX_' + node_name + 'sin'))
            onx_real_imag_part = ctx.make_node("Concat", inputs=[rng_cos.output[0], rng_sin.output[0]],
                                               name=utils.make_name('CPLX_' + node_name + '_cst_fft'),
                                               attr={'axis': 0})
            onx_real_imag_part_name = onx_real_imag_part.output[0]
            fft_length = None

        if axis != 0:
            shapei = list(np.arange(len(shape)))
            perm = shapei[:-2] + [shapei[-1], shapei[-2]]
            utils.make_sure(len(perm) >= 2, "perm cannot be empty.")
            trx = ctx.make_node(
                "Transpose", inputs=[input_name], attr=dict(perm=perm),
                name=utils.make_name(node_name + '_T_')).output[0]
        else:
            trx = input_name

        if axis is None:
            ctx.remove_node(node_name)
        mult = ctx.make_node(
            "MatMul", inputs=[onx_real_imag_part_name, trx],
            name=utils.make_name('CPLX_M_' + node_name + 'rfft'))

        if const_length:
            if axis == 1 or (fft_length < shape[1] and axis != 0):
                size = fft_length // 2 + 1
                new_shape = list(shape)
                new_shape[-2] = size
                if opset >= 10:
                    cst_axis = ctx.make_const(
                        name=utils.make_name('CPLX_csta'), np_val=np.array([-2], dtype=np.int64))
                    cst_zero = ctx.make_const(
                        name=utils.make_name('CPLX_cstz'), np_val=np.array([0], dtype=np.int64))
                    cst_length = ctx.make_const(
                        name=utils.make_name('CPLX_cstl'), np_val=np.array([size], dtype=np.int64))
                    sliced_mult = ctx.make_node(
                        "Slice", inputs=[mult.output[0], cst_zero.name, cst_length.name, cst_axis.name],
                        name=utils.make_name('CPLX_S_' + node_name + 'rfft'))
                else:
                    sliced_mult = ctx.make_node(
                        "Slice", inputs=[mult.output[0]], attr=dict(starts=[0], ends=[size], axes=[-2]),
                        name=utils.make_name('CPLX_S_' + node_name + 'rfft'))
            else:
                sliced_mult = mult
        else:
            utils.make_sure(
                False,
                "Dynamic length not fully implemented for dynamic fft_length or dynamic shape, fft_length=%r.",
                fft_length)

        if axis in (None, 1):
            new_shape = [2] + list(shape)
            shapei = list(np.arange(len(new_shape)))
            perm = shapei[:-2] + [shapei[-1], shapei[-2]]
            last_node = ctx.make_node(
                "Transpose", inputs=[sliced_mult.output[0]], attr=dict(perm=perm),
                name=utils.make_name('CPLX_T_' + node_name + 'rfft'))
        else:
            last_node = sliced_mult
        if axis is None:
            ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()
        return last_node


@tf_op("RFFT")
class RFFTOp(CommonFFTOp):
    # support more dtype

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        return cls.any_version(True, 1, ctx, node, **kwargs)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # Slice changed in opset 10.
        return cls.any_version(True, 10, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Unsqueeze changed in opset 13.
        return cls.any_version(True, 13, ctx, node, **kwargs)


@tf_op("FFT")
class FFTOp(CommonFFTOp):
    # support more dtype

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        return cls.any_version(False, 1, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        return cls.any_version(False, 13, ctx, node, **kwargs)


class CommonFFT2DOp(CommonFFTOp):

    @classmethod
    def any_version_2d(cls, const_length, opset, ctx, node, **kwargs):
        """
        Python code equivalent to FF2D (assuming fft_length[i] < input.shape[i] for all i).

        ::

            import numpy as np

            def _DFT_cst(N, fft_length, trunc=True):
                n = np.arange(N)
                k = n.reshape((N, 1)).astype(np.float64)
                M = np.exp(-2j * np.pi * k * n / fft_length)
                return M[:fft_length // 2 + 1] if trunc else M

            def DFT(x, fft_length=None, axis=1):
                if axis == 1:
                    x = x.T
                if fft_length is None:
                    fft_length = x.shape[0]
                cst = _DFT_cst(x.shape[0], fft_length, trunc=axis==1)
                if axis == 1:
                    return np.dot(cst, x).T
                else:
                    return np.dot(cst, x)

            def fft2d(mat, fft_length):
                mat = mat[:fft_length[0], :fft_length[1]]
                res = mat.copy()
                res = DFT(res, fft_length[1], axis=1)
                res = DFT(res, fft_length[0], axis=0)
                return res[:fft_length[0], :fft_length[1]//2 + 1]

        """
        consumers = ctx.find_output_consumers(node.output[0])
        consumer_types = set(op.type for op in consumers)
        utils.make_sure(
            consumer_types == {'ComplexAbs'},
            "Current implementation of RFFT2D only allows ComplexAbs as consumer not %r",
            consumer_types)

        input_name = node.input[0]
        onnx_dtype = ctx.get_dtype(input_name)
        utils.make_sure(onnx_dtype in CommonFFTOp.supported_dtypes, "Unsupported input type.")
        shape = ctx.get_shape(input_name)
        shape_n = shape[-1]

        if const_length:
            # RFFT: length of FFT is known, some computation
            # (see function make_dft_constant)
            # can be done at conversion time and stored as constant
            utils.make_sure(len(node.input) == 2, "Two inputs expected not %r", len(node.input))

            # This input should be a constant.
            fft_length_name = node.input[1]
            node_fft_length = ctx.get_node_by_output(fft_length_name, search_in_parent_graphs=True)
            utils.make_sure(node_fft_length.type == 'Const',
                            "fft_length should be a constant, the other case is not implemented yet.")
            value = node_fft_length.get_attr("value")
            value_array = to_array(value.t)
            utils.make_sure(value_array.shape == (2,),
                            "fft_length must be an array with two values not %r.", value_array) 
        else:
            raise NotImplementedError(
                "FFT2D with dynamic shape (known at execution) is not implemented yet.")

        # Slice
        if opset >= 10:
            cst_axis = ctx.make_const(
                name=utils.make_name('CPLX_csta'), np_val=np.array([-2, -1], dtype=np.int64))
            cst_zero = ctx.make_const(
                name=utils.make_name('CPLX_cstz'), np_val=np.array([0, 0], dtype=np.int64))
            cst_length = ctx.make_const(
                name=utils.make_name('CPLX_cstl'), np_val=np.array(value_array, dtype=np.int64))
            sliced_input_name = ctx.make_node(
                "Slice", inputs=[input_name, cst_zero.name, cst_length.name, cst_axis.name],
                name=utils.make_name('CPLX_S0_' + node.name))
        else:
            sliced_input_name = ctx.make_node(
                "Slice", inputs=[input_name], attr=dict(starts=[0, 0], ends=list(value_array), axes=[-2, -1]),
                name=utils.make_name('CPLX_S0_' + node.name))
        sliced_input_name = sliced_input_name.output[0]

        # First FFT
        last_node0 = cls.any_version(
            const_length, opset, ctx, None, axis=1, fft_length=value_array[1], dim=shape[1],
            onnx_dtype=onnx_dtype, shape=shape, input_name=sliced_input_name, **kwargs)
        last_node_name = last_node0.output[0]

        ind0 = ctx.make_const(name=utils.make_name('cst0'), np_val=np.array(0, dtype=np.int64))
        ind1 = ctx.make_const(name=utils.make_name('cst1'), np_val=np.array(1, dtype=np.int64))
        real_part = ctx.make_node(
            'Gather', inputs=[last_node_name, ind0.name], attr=dict(axis=0),
            name=utils.make_name('FFT2D_GReal_' + node.name))
        imag_part = ctx.make_node(
            'Gather', inputs=[last_node_name, ind1.name], attr=dict(axis=0),
            name=utils.make_name('FFT2D_GImag_' + node.name))

        real_node = cls.any_version(
            const_length, opset, ctx, None, axis=0, fft_length=value_array[0], dim=shape[0],
            onnx_dtype=onnx_dtype, shape=shape, input_name=real_part.output[0], **kwargs)
        imag_node = cls.any_version(
            const_length, opset, ctx, None, axis=0, fft_length=value_array[0], dim=shape[0],
            onnx_dtype=onnx_dtype, shape=shape, input_name=imag_part.output[0], **kwargs)

        # Extract real and imaginary parts, then applies the FFT in the other dimensions on each side.
        real_real_part = ctx.make_node(
            'Gather', inputs=[real_node.output[0], ind0.name], attr=dict(axis=0),
            name=utils.make_name('FFT2D_R_Real_' + node.name))
        real_imag_part = ctx.make_node(
            'Gather', inputs=[real_node.output[0], ind1.name], attr=dict(axis=0),
            name=utils.make_name('FFT2D_R_Imag_' + node.name))

        imag_real_part = ctx.make_node(
            'Gather', inputs=[imag_node.output[0], ind0.name], attr=dict(axis=0),
            name=utils.make_name('FFT2D_I_Real_' + node.name))
        imag_imag_part = ctx.make_node(
            'Gather', inputs=[imag_node.output[0], ind1.name], attr=dict(axis=0),
            name=utils.make_name('FFT2D_I_Imag_' + node.name))

        # Assemble all parts
        # w = a + ib
        # y1 = RFFT(a) = c + id, y2 = RFFT(b) = e + if
        # RFFT2D(a + ib)  -> c - f + i (d + e)

        new_real_node = ctx.make_node('Sub', inputs=[real_real_part.output[0], imag_imag_part.output[0]])
        new_imag_node = ctx.make_node('Add', inputs=[real_imag_part.output[0], imag_real_part.output[0]])
        if opset >= 13:
            ind0a = ctx.make_const(name=utils.make_name('cst0'), np_val=np.array([0], dtype=np.int64))
            angle_2d_real = ctx.make_node("Unsqueeze", inputs=[new_real_node.output[0], ind0a.name],
                                          name=utils.make_name('CPLX_' + node.name + 'angles2d'))
            angle_2d_imag = ctx.make_node("Unsqueeze", inputs=[new_imag_node.output[0], ind0a.name],
                                          name=utils.make_name('CPLX_' + node.name + 'angles2d'))
        else:
            angle_2d_real = ctx.make_node("Unsqueeze", inputs=[new_real_node.output[0]],
                                          name=utils.make_name('CPLX_' + node.name + 'angles2d'),
                                          attr={'axes': [0]})
            angle_2d_imag = ctx.make_node("Unsqueeze", inputs=[new_imag_node.output[0]],
                                          name=utils.make_name('CPLX_' + node.name + 'angles2d'),
                                          attr={'axes': [0]})

        last_node = ctx.make_node("Concat", inputs=[angle_2d_real.output[0], angle_2d_imag.output[0]],
                                  name=utils.make_name('CPLX_' + node.name + '_cst_fft2dc'),
                                  attr={'axis': 0})

        # FFT2D is different on tensorflow than on numpy.
        # shape may be different based on fft_length.

        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()
        ctx.remove_node(node.name)
        return last_node


@tf_op("RFFT2D")
class RFFT2DOp(CommonFFT2DOp):
    # support more dtype

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        return cls.any_version_2d(True, 1, ctx, node, **kwargs)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        # Slice changed in opset 10.
        return cls.any_version_2d(True, 10, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        # Unsqueeze changed in opset 13.
        return cls.any_version_2d(True, 13, ctx, node, **kwargs)


@tf_op("ComplexAbs")
class ComplexAbsOp:
    # support more dtype
    supported_dtypes = [
        onnx_pb.TensorProto.FLOAT,
        onnx_pb.TensorProto.FLOAT16,
        onnx_pb.TensorProto.DOUBLE,
        onnx_pb.TensorProto.COMPLEX64,
        onnx_pb.TensorProto.COMPLEX128,
    ]

    @classmethod
    def any_version(cls, opset, ctx, node, **kwargs):
        """
        Computes the modules of a complex.
        If the matrix dtype is not complex64 or complex128,
        it assumes the first dimension means real part (0)
        and imaginary part (1, :, :...).
        """
        onnx_dtype = ctx.get_dtype(node.input[0])
        if onnx_dtype is None:
            # This is not the best option. onnx_dtype is unknown when the Slice operator is used.
            # onnx<=1.9.0 fails at inferring the size.
            onnx_dtype = onnx_pb.TensorProto.FLOAT
        utils.make_sure(
            onnx_dtype in ComplexAbsOp.supported_dtypes, "Unsupported input type (node.name=%r, type=%r).",
            node.input[0], onnx_dtype)
        np_dtype = utils.map_onnx_to_numpy_type(onnx_dtype)
        shape = ctx.get_shape(node.input[0])
        if shape is not None:
            utils.make_sure(
                shape[0] == 2, "ComplexAbs expected the first dimension to be 2 but shape is %r", shape)

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
            name=utils.make_name('ComplexAbs' + node.name))

        ctx.replace_all_inputs(node.output[0], last_node.output[0])  # ops=ctx.get_nodes()

    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        cls.any_version(1, ctx, node, **kwargs)

    @classmethod
    def version_13(cls, ctx, node, **kwargs):
        cls.any_version(13, ctx, node, **kwargs)
