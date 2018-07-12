"""
A simple example how to call tensorflow-onnx via python.
"""

import tensorflow as tf
import tf2onnx

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph)
    model_proto = onnx_graph.make_model("test", ["input:0"], ["output:0"])
    with open("/tmp/model.onnx", "wb") as f:
        f.write(model_proto.SerializeToString())
