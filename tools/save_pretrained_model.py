# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
Save pre-trained model.
"""
import tensorflow as tf
import numpy as np

# pylint: disable=redefined-outer-name,reimported

def save_pretrained_model(sess, outputs, feeds, out_dir, model_name="pretrained"):
    """Save pretrained model and config"""
    try:
        import os
        import sys
        import tensorflow as tf
        import subprocess
        to_onnx_path = "{}/to_onnx".format(out_dir)
        if not os.path.isdir(to_onnx_path):
            os.makedirs(to_onnx_path)
        saved_model = "{}/saved_model".format(to_onnx_path)
        inputs_path = "{}/inputs.npy".format(to_onnx_path)
        pretrained_model_yaml_path = "{}/pretrained.yaml".format(to_onnx_path)
        envars_path = "{}/environment.txt".format(to_onnx_path)
        pip_requirement_path = "{}/requirements.txt".format(to_onnx_path)

        print("===============Save Saved Model========================")
        if os.path.exists(saved_model):
            print("{} already exists, SKIP".format(saved_model))
            return

        print("Save tf version, python version and installed packages")
        tf_version = tf.__version__
        py_version = sys.version
        pip_packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"])
        pip_packages = pip_packages.decode("UTF-8")
        with open(envars_path, "w") as fp:
            fp.write(tf_version + os.linesep)
            fp.write(py_version)
        with open(pip_requirement_path, "w") as fp:
            fp.write(pip_packages)

        print("Save model for tf2onnx: {}".format(to_onnx_path))
        # save inputs
        inputs = {}
        for inp, value in feeds.items():
            if isinstance(inp, str):
                inputs[inp] = value
            else:
                inputs[inp.name] = value
        np.save(inputs_path, inputs)
        print("Saved inputs to {}".format(inputs_path))

        # save graph and weights
        from tensorflow.saved_model import simple_save
        simple_save(sess, saved_model,
                    {n: i for n, i in zip(inputs.keys(), feeds.keys())},
                    {op.name: op for op in outputs})
        print("Saved model to {}".format(saved_model))

        # generate config
        pretrained_model_yaml = '''
{}:
  model: ./saved_model
  model_type: saved_model
  input_get: get_ramp
'''.format(model_name)
        pretrained_model_yaml += "  inputs:\n"
        for inp, _ in inputs.items():
            pretrained_model_yaml += \
                "    \"{input}\": np.array(np.load(\"./inputs.npy\")[()][\"{input}\"])\n".format(input=inp)
        outputs = [op.name for op in outputs]
        pretrained_model_yaml += "  outputs:\n"
        for out in outputs:
            pretrained_model_yaml += "    - {}\n".format(out)
        with open(pretrained_model_yaml_path, "w") as f:
            f.write(pretrained_model_yaml)
        print("Saved pretrained model yaml to {}".format(pretrained_model_yaml_path))
        print("=========================================================")
    except Exception as ex:  # pylint: disable=broad-except
        print("Error: {}".format(ex))


def test():
    """Test sample."""
    x_val = np.random.rand(5, 20).astype(np.float32)
    y_val = np.random.rand(20, 10).astype(np.float32)
    x = tf.placeholder(tf.float32, x_val.shape, name="x")
    y = tf.placeholder(tf.float32, y_val.shape, name="y")
    z = tf.matmul(x, y)
    w = tf.get_variable("weight", [5, 10], dtype=tf.float32)
    init = tf.global_variables_initializer()
    outputs = [z + w]
    feeds = {x: x_val, y: y_val}
    with tf.Session() as sess:
        sess.run(init)
        sess.run(outputs, feeds)
        # NOTE: NOT override the saved model, so put below snippet after testing the BEST model.
        # if you perform testing several times.
        save_pretrained_model(sess, outputs, feeds, "./tests", model_name="test")


if __name__ == "__main__":
    test()
