# SPDX-License-Identifier: Apache-2.0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import numpy
from onnxruntime import InferenceSession
from _tools import generate_random_images, benchmark, measure_time
import tensorflow as tf
import tensorflow_hub as hub


def main(opset=13):
    url = "https://tfhub.dev/google/humpback_whale/1?tf-hub-format=compressed"
    dest = "tf-humpback-whale"
    name = "humpback-whale"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))
    print("[download data]")
    FILENAME = 'gs://bioacoustics-www1/sounds/Cross_02_060203_071428.d20_7.wav'
    pkl_name = os.path.join(dest, "data.pkl")
    if not os.path.exists(pkl_name):
        with open(pkl_name, "wb") as f:
            waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(FILENAME))
            waveform = tf.expand_dims(waveform, 0)  # makes a batch of size 1
            context_step_samples = tf.cast(sample_rate, tf.int64)
            data = dict(waveform=waveform, context_step_samples=context_step_samples)
            pickle.dump(data, f)
    else:
        with open(pkl_name, "rb") as f:
            data = pickle.load(f)
        waveform = data["waveform"]
        context_step_samples = data["context_step_samples"]
    print("[data] done. context_step_samples=", context_step_samples.numpy())

    def benchmark_custom(local_name):
        model = hub.load(local_name)
        score_fn = model.signatures['score']
        scores = score_fn(waveform=waveform, context_step_samples=context_step_samples)
        imgs_tf = [dict(waveform=waveform, context_step_samples=context_step_samples)]
        results_tf, duration_tf = measure_time(
            lambda inputs: score_fn(**inputs), imgs_tf)
        return scores, results_tf, duration_tf

    imgs = generate_random_images(shape=(1, 750000, 1), scale=1., n=2)
    inputs = [dict(waveform=waveform.numpy(),
                   context_step_samples=numpy.array(
                    context_step_samples.numpy(), dtype=numpy.int64))]
    benchmark(url, dest, onnx_name, opset, inputs, optimize=False,
              signature='score', custom_tf=benchmark_custom)


if __name__ == "__main__":
    with tf.device('/CPU:0'):
        main()
