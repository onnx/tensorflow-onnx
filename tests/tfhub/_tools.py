# SPDX-License-Identifier: Apache-2.0

"""
The following code compares the speed of tensorflow against onnxruntime
with a model downloaded from Tensorflow Hub.
"""
import os
import sys
import time
import tarfile
import zipfile
import subprocess
import datetime
import numpy
from tqdm import tqdm
import onnxruntime


def generate_random_images(shape=(1, 100, 100, 3), n=10, dtype=numpy.float32):
    imgs = []
    for i in range(n):
        sh = shape
        img = numpy.clip(numpy.abs(numpy.random.randn(*sh)), 0, 1) * 255
        img = img.astype(dtype)
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
        from tf2onnx import utils
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


def download_tflite(url, dest, verbose=True):
    """
    Downloads a model from tfhub.
    The function assumes the format is `.tflite`.
    """
    if not os.path.exists(dest):
        os.makedirs(dest)
    fpath = os.path.join(dest, "model.tflite")
    if not os.path.exists(fpath):
        from tf2onnx import utils
        if verbose:
            print("Download %r." % fpath)
        utils.get_url(url, fpath)
    return fpath


def convert_model(model_name, output_path, opset=13, verbose=True):
    """
    Converts the downloaded model into ONNX.
    """
    if not os.path.exists(output_path):
        begin = datetime.datetime.now()
        cmdl = ['-m', 'tf2onnx.convert', '--saved-model',
                '"%s"' % os.path.abspath(model_name).replace("\\", "/"),
                '--output', '"%s"' % os.path.abspath(output_path).replace("\\", "/"),
                '--opset', "%d" % opset]
        if verbose:
            print("cmd: python %s" % " ".join(cmdl))
        pproc = subprocess.Popen(
            cmdl, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            executable=sys.executable.replace("pythonw", "python"))
        stdoutdata, stderrdata = pproc.communicate()
        if verbose:
            print('--OUT--')
            print(stdoutdata.decode('ascii'))
            print('--ERR--')
            print(stderrdata.decode('ascii'))
            print("Duration %r." % (datetime.datetime.now() - begin))


def convert_tflite(model_name, output_path, opset=13, verbose=True):
    """
    Converts the downloaded model into ONNX.
    """
    if not os.path.exists(output_path):
        begin = datetime.datetime.now()
        cmdl = ['-m', 'tf2onnx.convert', '--tflite',
                '"%s"' % os.path.abspath(model_name).replace("\\", "/"),
                '--output', '"%s"' % os.path.abspath(output_path).replace("\\", "/"),
                '--opset', "%d" % opset]
        if verbose:
            print("cmd: python %s" % " ".join(cmdl))
        pproc = subprocess.Popen(
            cmdl, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            executable=sys.executable.replace("pythonw", "python"))
        stdoutdata, stderrdata = pproc.communicate()
        if verbose:
            print('--OUT--')
            print(stdoutdata.decode('ascii'))
            print('--ERR--')
            print(stderrdata.decode('ascii'))
            print("Duration %r." % (datetime.datetime.now() - begin))


def check_discrepencies(out1, out2, threshold=1e-3):
    """
    Compares two tensors. Raises an exception if it fails.
    """
    if out1.dtype != out2.dtype:
        raise AssertionError("Type mismatch %r != %r." % (out1.dtype, out2.dtype))
    if out1.shape != out2.shape:
        raise AssertionError("Shape mismatch %r != %r." % (out1.shape, out2.shape))
    diff = numpy.abs(out1.ravel() - out2.ravel()).max()
    if diff > threshold:
        raise AssertionError("Discrependcies %r > %r." % (diff, threshold))


def benchmark(url, dest, onnx_name, opset, imgs, verbose=True, threshold=1e-3,
              signature=None):
    """
    Runs a simple benchmark.
    Goes through every steps (download, convert).
    Skips them if already done.
    """
    fpath, tname = download_model(url, dest)
    if verbose:
        print("Created %r, %r." % (fpath, tname))

    # Converts the model.
    if verbose:
        print("Convert model in %r." % dest)
    convert_model(tname, onnx_name, opset)
    if verbose:
        print("Created %r." % onnx_name)

    # Benchmarks both models.
    ort = onnxruntime.InferenceSession(onnx_name)

    if verbose:
        print("ONNX inputs:")
        for a in ort.get_inputs():
            print("  {}: {}, {}".format(a.name, a.type, a.shape))
        print("ONNX outputs:")
        for a in ort.get_outputs():
            print("  {}: {}, {}".format(a.name, a.type, a.shape))

    # onnxruntime
    input_name = ort.get_inputs()[0].name
    fct_ort = lambda img: ort.run(None, {input_name: img})[0]
    results_ort, duration_ort = measure_time(fct_ort, imgs)
    if verbose:
        print("ORT", len(imgs), duration_ort)

    # tensorflow
    import tensorflow_hub as hub
    from tensorflow import convert_to_tensor
    model = hub.load(url.split("?")[0])
    if signature is not None:
        model = model.signatures['serving_default']
    imgs_tf = [convert_to_tensor(img) for img in imgs]
    results_tf, duration_tf = measure_time(model, imgs_tf)

    if verbose:
        print("TF", len(imgs), duration_tf)
        mean_ort = sum(duration_ort) / len(duration_ort)
        mean_tf = sum(duration_tf) / len(duration_tf)
        print("ratio ORT=%r / TF=%r = %r" % (mean_ort, mean_tf, mean_ort / mean_tf))

    # checks discrepencies
    res = model(imgs_tf[0])
    if isinstance(res, dict):
        if len(res) != 1:
            raise NotImplementedError("TF output contains more than one output: %r." % res)
        output_name = ort.get_outputs()[0].name
        if output_name not in res:
            raise AssertionError("Unable to find output %r in %r." % (output_name, list(sorted(res))))
        res = res[output_name]
    check_discrepencies(fct_ort(imgs[0]), res.numpy(), threshold)
    return duration_ort, duration_tf


def benchmark_tflite(url, dest, onnx_name, opset, imgs, verbose=True, threshold=1e-3):
    """
    Runs a simple benchmark with a tflite model.
    Goes through every steps (download, convert).
    Skips them if already done.
    """
    tname = download_tflite(url, dest)
    if verbose:
        print("Created %r." % tname)

    # Converts the model.
    if verbose:
        print("Convert model in %r." % dest)
    convert_tflite(tname, onnx_name, opset)
    if verbose:
        print("Created %r." % onnx_name)

    # Benchmarks both models.
    ort = onnxruntime.InferenceSession(onnx_name)

    if verbose:
        print("ONNX inputs:")
        for a in ort.get_inputs():
            print("  {}: {}, {}".format(a.name, a.type, a.shape))
        print("ONNX outputs:")
        for a in ort.get_outputs():
            print("  {}: {}, {}".format(a.name, a.type, a.shape))

    # onnxruntime
    input_name = ort.get_inputs()[0].name
    fct_ort = lambda img: ort.run(None, {input_name: img})[0]
    results_ort, duration_ort = measure_time(fct_ort, imgs)
    if verbose:
        print("ORT", len(imgs), duration_ort)

    # tensorflow
    import tensorflow_hub as hub
    from tensorflow import convert_to_tensor
    model = hub.load(url.split("?")[0])
    if signature is not None:
        model = model.signatures['serving_default']
    imgs_tf = [convert_to_tensor(img) for img in imgs]
    results_tf, duration_tf = measure_time(model, imgs_tf)

    if verbose:
        print("TF", len(imgs), duration_tf)
        mean_ort = sum(duration_ort) / len(duration_ort)
        mean_tf = sum(duration_tf) / len(duration_tf)
        print("ratio ORT=%r / TF=%r = %r" % (mean_ort, mean_tf, mean_ort / mean_tf))

    # checks discrepencies
    res = model(imgs_tf[0])
    if isinstance(res, dict):
        if len(res) != 1:
            raise NotImplementedError("TF output contains more than one output: %r." % res)
        output_name = ort.get_outputs()[0].name
        if output_name not in res:
            raise AssertionError("Unable to find output %r in %r." % (output_name, list(sorted(res))))
        res = res[output_name]
    check_discrepencies(fct_ort(imgs[0]), res.numpy(), threshold)
    return duration_ort, duration_tf