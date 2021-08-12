# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from onnxruntime import InferenceSession
from _tools import generate_random_images, benchmark


def main(opset=13):
    url = "https://tfhub.dev/google/humpback_whale/1?tf-hub-format=compressed"
    dest = "tf-humpback-whale"
    name = "humpback-whale"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    kind = "cmd"
    if kind == "function":
        import tensorflow as tf
        import tensorflow_hub as hub
        import tf2onnx
        model = hub.load('https://tfhub.dev/google/humpback_whale/1')
        FILENAME = 'gs://bioacoustics-www1/sounds/Cross_02_060203_071428.d20_7.wav'
        waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(FILENAME))
        waveform = tf.expand_dims(waveform, 0)  # makes a batch of size 1
        context_step_samples = tf.cast(sample_rate, tf.int64)
        print(waveform.dtype, waveform.shape, sample_rate.dtype, sample_rate.shape, sample_rate)
        
        spec = (tf.TensorSpec((None, ) + waveform.shape[-2:], tf.float32, name="waveform"),
                tf.TensorSpec((1, 1), tf.int64, name="context_step_samples"))
        inputs = {'waveform': waveform.numpy(),
                  'context_step_samples': context_step_samples.numpy()}
                
        tf2onnx.convert.from_function(
            model.signatures['score'], input_signature=spec, opset=13, output_path=onnx_name)
        # AttributeError: '_WrapperFunction' object has no attribute 'get_concrete_function'

        sess = InferenceSession(onnx_name)
        got = sess.run(None, inputs)
        print(got)
        
        score_fn = model.signatures['score']
        scores = score_fn(waveform=waveform, context_step_samples=context_step_samples)
    
    if kind == "keras":
        import tensorflow as tf
        import tensorflow_hub as hub
        import tf2onnx
        model = hub.load('https://tfhub.dev/google/humpback_whale/1').model
        FILENAME = 'gs://bioacoustics-www1/sounds/Cross_02_060203_071428.d20_7.wav'
        waveform, sample_rate = tf.audio.decode_wav(tf.io.read_file(FILENAME))
        waveform = tf.expand_dims(waveform, 0)  # makes a batch of size 1
        context_step_samples = tf.cast(sample_rate, tf.int64)
        print(waveform.dtype, waveform.shape, sample_rate.dtype, sample_rate.shape, sample_rate)

        spec = (tf.TensorSpec((None, ) + waveform.shape[-2:], tf.float32, name="waveform"),
                tf.TensorSpec((1, 1), tf.int64, name="context_step_samples"))
        inputs = {'waveform': waveform.numpy(),
                  'context_step_samples': context_step_samples.numpy()}
                
        tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=onnx_name)
        # AttributeError: '_UserObject' object has no attribute 'output_names'

        sess = InferenceSession(onnx_name)
        got = sess.run(None, inputs)
        print(got)
        
        score_fn = model.signatures['score']
        scores = score_fn(waveform=waveform, context_step_samples=context_step_samples)

    if kind == 'cmd':
        imgs = generate_random_images(shape=(1, 10000, 1), scale=1.)
        inputs = [dict(waveform=img,
                       context_step_samples=numpy.array(512, dtype=numpy.int64))
                  for img in imgs]
        benchmark(url, dest, onnx_name, opset, inputs, optimize=False,
                  signature='score')
        # onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: 
        # [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node. Name:'StatefulPartitionedCall/Reshape_1' Status Message: C:\xadupre\microsoft_xadupre\onnxruntime\onnxruntime\core\providers\cpu\tensor\reshape_helper.h:42 onnxruntime::ReshapeHelper::ReshapeHelper gsl::narrow_cast<int64_t>(input_shape.Size()) == size was false. The input tensor cannot be reshaped to the requested shape. 
        # Input shape:{0,1}, requested shape:{1,1,1}


if __name__ == "__main__":
    main()
