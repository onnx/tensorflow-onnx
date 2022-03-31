# SPDX-License-Identifier: Apache-2.0

# coding: utf-8
"""
Profiles the conversion of a Keras model.
"""
import sys
import cProfile
from pstats import SortKey, Stats
import io
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, EfficientNetB2
from tf2onnx import tfonnx
try:
    from pyinstrument import Profiler
except ImportError:
    Profiler = None


def spy_model(name):
    "Creates the model."
    with tf.compat.v1.Session(graph=tf.Graph()) as session:
        if name == "MobileNet":
            model = MobileNet()
        elif name == "EfficientNetB2":
            model = EfficientNetB2()
        else:
            raise ValueError("Unknown model name %r." % name)

        graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=session,
            input_graph_def=session.graph_def,
            output_node_names=[model.output.op.name])

    return graph_def, model


def spy_convert(graph_def, model):
    "Converts the model."
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def=graph_def, name='')

        def spy_convert_in():
            return tfonnx.process_tf_graph(
                tf_graph=graph, input_names=[model.input.name],
                output_names=[model.output.name])

        spy_convert_in()


def create(name):
    "Creates the model."
    graph_def, model = spy_model(name)
    return graph_def, model


def convert(graph_def, model):
    "Converts the model."
    spy_convert(graph_def, model)


def profile(profiler="none", name="MobileNet", show_all=False):
    """
    Profiles the conversion of a model.

    :param profiler: one among none, spy, pyinstrument, cProfile
    :param name: model to profile, MobileNet, EfficientNetB2
    :param showall: used by pyinstrument to show all functions
    """
    print("create(%r, %r)" % (profiler, name))
    graph_def, model = create(name)
    print("profile(%r, %r)" % (profiler, name))
    if profiler == "none":
        convert(graph_def, model)
    elif profiler == "spy":
        # py-spy record -r 10 -o profile.svg -- python conversion_time.py spy
        convert(graph_def, model)
    elif profiler == "pyinstrument":
        if Profiler is None:
            raise ImportError("pyinstrument is not installed")
        profiler = Profiler(interval=0.0001)
        profiler.start()
        convert(graph_def, model)
        profiler.stop()
        print(profiler.output_text(unicode=False, color=False, show_all=show_all))
    elif profiler == "cProfile":
        pr = cProfile.Profile()
        pr.enable()
        convert(graph_def, model)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
    else:
        raise ValueError("Unknown profiler %r." % profiler)


def main(args):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--profiler', default='none',
                        choices=['none', 'spy', 'pyinstrument', 'cProfile'],
                        help='a profiler')
    parser.add_argument('--name', default="MobileNet",
                        choices=['MobileNet', 'EfficientNetB2'],
                        help="a model")
    parser.add_argument('--showall', type=bool, default=False,
                        help="used by pyinstrument to show all functions")
    res = parser.parse_args(args)
    profile(res.profiler, res.name, res.showall)


if __name__ == '__main__':
    print('Begin profiling with', sys.argv[1:])
    main(sys.argv[1:])
    print('Profile complete.')
