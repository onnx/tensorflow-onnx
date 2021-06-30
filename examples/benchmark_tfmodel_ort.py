# SPDX-License-Identifier: Apache-2.0

"""
The following code compares the speed of tensorflow against onnxruntime
with a model downloaded from Tensorflow Hub.
"""
import os
import sys
import time
import tarfile
import subprocess
import datetime
import numpy
from tqdm import tqdm
import tensorflow_hub as hub
import onnxruntime as ort
from tf2onnx import utils


def generate_random_images(shape=(100, 100), n=10):
    imgs = []
    for i in range(n):
        sh = (1,) + shape + (3,)
        img = numpy.clip(numpy.abs(numpy.random.randn(*sh)), 0, 1) * 255
        img = img.astype(numpy.float32)
        imgs.append(img)
    return imgs


def measure_time(fct, imgs, n=50, timeout=15):
    """
    Runs *n* times the same function taking one parameter
    from *imgs*. It stops if the total time overcomes *timeout*.
    It also runs once the function before measuring.
    """
    # Let's run it once first.
    fct(imgs[0])
    # The time is measured for n iterations except if the total time
    # overcomes timeout.
    results = []
    times = []
    for i in tqdm(range(0, n)):
        img = imgs[i % len(imgs)]
        begin = time.perf_counter()
        result = fct(img)
        end = time.perf_counter()
        results.append(result)
        times.append(end - begin)
        if sum(times) > timeout:
            break
    return results, times


def download_model(url, dest, verbose=True):
    """
    Downloads a model from tfhub and unzips it.
    The function assumes the format is `.tar.gz`.
    """
    if not os.path.exists(dest):
        os.makedirs(dest)
    fpath = os.path.join(dest, "model.tar.gz")
    if not os.path.exists(fpath):
        if verbose:
            print("Download %r." % fpath)
        utils.get_url(url, fpath)
    tname = os.path.join(dest, "model_path")
    if not os.path.exists(tname):
        if verbose:
            print("Untar %r." % tname)
        tar = tarfile.open(fpath)
        tar.extractall(tname)
        tar.close()        
    return fpath, tname


def convert_model(model_name, output_path, opset=13, verbose=True):
    """
    Converts the downloaded model into ONNX.
    """
    if not os.path.exists(output_path):
        begin = datetime.datetime.now()
        cmdl = ['-m', 'tf2onnx.convert', '--saved-model',
                '"%s"' % model_name.replace("\\", "/"),
                '--output', '"%s"' % output_path.replace("\\", "/"),
                '--opset', "%d" % opset]
        if verbose:
            print("cmd: python %s" % " ".join(cmdl))
        pproc = subprocess.Popen(cmdl, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 executable=sys.executable.replace("pythonw", "python"))
        stdoutdata, stderrdata = pproc.communicate()
        if verbose:
            print('--OUT--')
            print(stdoutdata)
            print('--ERR--')
            print(stderrdata)
            print("Duration %r." % (datetime.datetime.now() - begin))


# Downloads the model
url = "https://tfhub.dev/captain-pool/esrgan-tf2/1?tf-hub-format=compressed"
dest = os.path.abspath("tf-esrgan-tf2")
name = "esrgan-tf2"
opset = 13
onnx_name = os.path.join(dest, "esrgan-tf2-%d.onnx" % opset)

fpath, tname = download_model(url, dest)
print("Created %r, %r." % (fpath, tname))

# Converts the model.
print("Convert model in %r." % dest)
convert_model(tname, onnx_name, opset)
print("Created %r." % onnx_name)

# Generates random images.
print("Generates images.")
imgs = generate_random_images()

# Benchmarks both models.
ort = ort.InferenceSession(onnx_name)
fct_ort = lambda img: ort.run(None, {'input_0': img})
results_ort, duration_ort = measure_time(fct_ort, imgs)
print("ORT", len(imgs), duration_ort)

model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
results_tf, duration_tf = measure_time(model, imgs)
print("TF", len(imgs), duration_tf)

mean_ort = sum(duration_ort) / len(duration_ort)
mean_tf = sum(duration_tf) / len(duration_tf)
print("ratio ORT=%r / TF=%r = %r" % (mean_ort, mean_tf, mean_ort / mean_tf))
