# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from _tools import generate_random_images, benchmark, measure_time
from tensorflow import convert_to_tensor
import tensorflow as tf
import tensorflow_hub as hub
import tf2onnx


def main(opset=13):
    print('[begin]')
    url = "https://tfhub.dev/deepmind/enformer/1?tf-hub-format=compressed"
    dest = "tf-enformer"
    name = "enformer"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    model = None
    if not os.path.exists(onnx_name):
        if model is None:
            model = hub.load("https://tfhub.dev/deepmind/enformer/1").model

        tf2onnx.convert.from_function(
            model.predict_on_batch,
            [tf.TensorSpec([None, 393216, 4], tf.float32)],
            opset=13, output_path=onnx_name)

    # benchmark(url, dest, onnx_name, opset, imgs)
    print("[generate dummy images]")
    imgs = generate_random_images(shape=(1, 393216, 4), scale=0.)

    ort = InferenceSession(onnx_name)
    fct_ort = lambda img: ort.run(None, {'args_0': img})[0]

    if model is None:
        model = hub.load("https://tfhub.dev/deepmind/enformer/1").model

    fct_tf = lambda img: model.predict_on_batch(img)

    print('[benchmark tf]')
    imgs_tf = [convert_to_tensor(img) for img in imgs]
    results_tf, duration_tf = measure_time(fct_tf, imgs)
    print("TF", len(imgs), duration_tf)

    print('[benchmark ort]')
    results_ort, duration_ort = measure_time(fct_ort, imgs)
    print("ORT", len(imgs), duration_ort)    

    mean_ort = sum(duration_ort) / len(duration_ort)
    mean_tf = sum(duration_tf) / len(duration_tf)
    print("ratio ORT=%r / TF=%r = %r" % (mean_ort, mean_tf, mean_ort / mean_tf))

    # discrepencies
    assert_almost_equal(results_tf[0]['human'], results_ort[0], decimal=4)
    print('[end]')

if __name__ == "__main__":
    main()
