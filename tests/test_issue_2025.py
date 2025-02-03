# SPDX-License-Identifier: Apache-2.0
# noqa: C0415, C0114, C0115
import unittest
from backend_test_base import Tf2OnnxBackendTestBase


class TestIssue2025(Tf2OnnxBackendTestBase):
    def test_tanhgrad(self):

        import tensorflow as tf
        import tf2onnx
        from tf2onnx.handler import tf_op
        import numpy as np

        @tf_op("TanhGrad")
        class TanhGrad:
            @classmethod
            def version_1(cls, ctx, node, **kwargs):
                tanh_output = node.input[0]
                grad = node.input[1]
                square = ctx.make_node("Mul", [tanh_output, tanh_output])
                one = ctx.make_const(
                    name=node.name + "_one", np_val=np.array(1, dtype=np.float32)
                )
                derivative = ctx.make_node("Sub", [one.output[0], square.output[0]])
                result = ctx.make_node("Mul", [derivative.output[0], grad])
                ctx.replace_all_inputs(node.output[0], result.output[0])
                return result.output

        class QFGrad(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.output_names = ["grad"]

            def calc_q_grad(self, x):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = tf.keras.activations.tanh(tf.abs(x))
                # tf.raw_ops.TanhGrad
                x_grad = tape.gradient(y, x)
                return x_grad

            def call(self, x):
                q_grad = self.calc_q_grad(x)
                return q_grad

        model = QFGrad()
        x = tf.random.uniform((1, 1, 6))
        model(x)

        save_path = "test_tanhgrad.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(
            model=model,
            input_signature=(tf.TensorSpec((1, 1, 6), dtype=tf.float32, name="x"),),
            opset=13,
            output_path=save_path,
        )
        node_types = [n.op_type for n in model_proto.graph.node]
        self.assertNotIn("TanhGrad", node_types)



if __name__ == "__main__":
    unittest.main()
