# SPDX-License-Identifier: Apache-2.0
import os
import numpy
from _tools import generate_random_images, benchmark_tflite


def main(opset=13):
    url = "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
    dest = "tf-yamnet-tflite"
    name = "yamnet"
    onnx_name = os.path.join(dest, "%s-%d.onnx" % (name, opset))

    imgs = generate_random_images(shape=(15600, ), dtype=numpy.float32, scale=0.)
    
    benchmark_tflite(url, dest, onnx_name, opset, imgs, names=[
        ('stft/rfft3', 'FFT_stft/rfft4_reshape__190:0'),
        ('magnitude_spectrogram', 'ComplexAbsmagnitude_spectrogram__206:0'),
        ('log_mel_spectrogram', 'log_mel_spectrogram'),
    ])


if __name__ == "__main__":
    main()
