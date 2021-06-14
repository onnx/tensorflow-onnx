# SPDX-License-Identifier: Apache-2.0

import os
import tensorflow as _tf

from distutils.version import StrictVersion

is_tf2 = StrictVersion(_tf.__version__.split('-')[0]) >= StrictVersion('2.0.0')


def normalize_tensor_shape(tensor_shape):
    if is_tf2:
        return [d for d in tensor_shape]
    else:
        return [d.value for d in tensor_shape]


def dump_graph_into_tensorboard(tf_graph):
    # type: (_tf.Graph) -> None
    _tb_log_dir = os.environ.get('TB_LOG_DIR')
    if _tb_log_dir:
        if is_tf2:
            from tensorflow.python.ops.summary_ops_v2 import graph as write_graph
            pb_visual_writer = _tf.summary.create_file_writer(_tb_log_dir)
            with pb_visual_writer.as_default():
                write_graph(tf_graph)
        else:
            from tensorflow.python.summary import summary
            pb_visual_writer = summary.FileWriter(_tb_log_dir)
            pb_visual_writer.add_graph(tf_graph)


if is_tf2:
    tensorflow = _tf.compat.v1

    def is_subclassed(layer):
        """Returns True if the object is a subclassed layer or subclassed model."""
        return (layer.__module__.find('keras.engine') == -1 and
                layer.__module__.find('keras.layers') == -1)
else:
    tensorflow = _tf

    def is_subclassed(layer):
        return False
