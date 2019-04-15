import tensorflow as tf
import numpy as np

def save_pretrained_model(sess, outputs, feeds, out_dir, model_name="pretrained"):
    try:
        import os
        to_onnx_path = "{}/to_onnx".format(out_dir)
        if not os.path.isdir(to_onnx_path):
            os.makedirs(to_onnx_path)
        saved_model = "{}/saved_model".format(to_onnx_path)
        inputs_path = "{}/inputs.npy".format(to_onnx_path)
        pretrained_model_yaml_path = "{}/pretrained.yaml".format(to_onnx_path)

        print("===============Save Frozen Graph========================")
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
                    {n: i for n,i in zip(inputs.keys(), feeds.keys())},
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
            pretrained_model_yaml += "    \"{}\": np.array(np.load(\"./inputs.npy\")[()][\"{}\"])\n".format(
                inp, inp
            )
        outputs = [op.name for op in outputs]
        pretrained_model_yaml += "  outputs:\n"
        for out in outputs:
            pretrained_model_yaml += "    - {}\n".format(out)
        with open(pretrained_model_yaml_path, "w") as f:
            f.write(pretrained_model_yaml)
        print("Saved pretrained model yaml to {}".format(pretrained_model_yaml_path))
        print("=========================================================")
    except Exception as ex:
        print("Error: {}".format(ex))


def test():
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
        out = sess.run(outputs, feeds)
        # NOTE: Put below snippet after the LAST testing step
        save_pretrained_model(sess, outputs, feeds, "./tests", model_name="test")


if __name__ == "__main__":
    test()
