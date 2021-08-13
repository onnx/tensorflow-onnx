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
from collections import OrderedDict
import numpy
from tqdm import tqdm
import onnxruntime


def generate_random_images(shape=(1, 100, 100, 3), n=10, dtype=numpy.float32, scale=255):
    imgs = []
    for i in range(n):
        sh = shape
        img = numpy.clip(numpy.abs(numpy.random.randn(*sh)), 0, 1) * scale
        img = img.astype(dtype)
        imgs.append(img)
    return imgs


def generate_text_inputs():
    """
    preprocessor = hub.load("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
    encoder = hub.load("https://tfhub.dev/tensorflow/albert_en_xlarge/3")
    sentences = tf.constant(["Hi I'm some text"])
    embedded_inputs = {k: v.numpy() for k, v in preprocessor(sentences).items()}
    """
    one = OrderedDict([
        ('input_word_ids', numpy.array([[
            2, 4148, 31, 22, 79, 109, 1854, 3, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,0]]).reshape((1, -1))),
        ('input_type_ids', numpy.array([[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape((1, -1))),
        ('input_mask', numpy.array([[
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape((1, -1)))])
    return [one for i in range(10)]


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


def convert_model(model_name, output_path, opset=13, tag=None, signature=None, verbose=True):
    """
    Converts the downloaded model into ONNX.
    """
    ext = os.path.splitext(output_path)[-1]
    large_model = ext == ".zip"
    if not os.path.exists(output_path):
        begin = datetime.datetime.now()
        cmdl = ['python', '-m', 'tf2onnx.convert', '--saved-model',
                '%s' % os.path.abspath(model_name).replace("\\", "/"),
                '--output', '%s' % os.path.abspath(output_path).replace("\\", "/"),
                '--opset', "%d" % opset]
        if signature is not None:
            cmdl.append('--signature_def=%s' % signature)
        if tag is not None:
            cmdl.append('--tag=%s' % tag)
        if large_model:
            cmdl.append('--large_model')
        if verbose:
            print("cmd: %s" % " ".join(cmdl))
        pproc = subprocess.Popen(
            cmdl, shell=False, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            executable=None)
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
        cmdl = ['python', '-m', 'tf2onnx.convert', '--tflite',
                '"%s"' % os.path.abspath(model_name).replace("\\", "/"),
                '--output', '"%s"' % os.path.abspath(output_path).replace("\\", "/"),
                '--opset', "%d" % opset]
        if verbose:
            print("cmd: %s" % " ".join(cmdl))
        pproc = subprocess.Popen(
            cmdl, shell=False, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            executable=None)
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
    if isinstance(out1, list):
        if len(out1) > 1:
            if len(out1) != len(out2):
                raise AssertionError(
                    "Mismatched number of outputs, %d for ONNX, %d for TF." % (
                        len(out1), len(out2)))
            for i, (a, b) in enumerate(zip(out1, out2)):
                try:
                    check_discrepencies(out1[i], out2[i].numpy(), threshold=1e-3)
                except AssertionError as e:
                    raise AssertionError("Discrepency with output %d." % i) from e
            return
        else:
            out1 = out1[0]
    if out1.dtype != out2.dtype:
        raise AssertionError("Type mismatch %r != %r." % (out1.dtype, out2.dtype))
    if out1.shape != out2.shape:
        raise AssertionError("Shape mismatch %r != %r." % (out1.shape, out2.shape))
    diff = numpy.abs(out1.ravel() - out2.ravel()).max()
    if diff > threshold:
        raise AssertionError("Discrependcies %r > %r." % (diff, threshold))


def benchmark(url, dest, onnx_name, opset, imgs, verbose=True, threshold=1e-3,
              signature=None, tag=None, output_name=None, ort_name=None,
              optimize=True, convert_tflite=None, custom_tf=None):
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
    convert_model(tname, onnx_name, opset, tag=tag, signature=signature)
    if verbose:
        print("Created %r." % onnx_name)

    # unzip large_model
    ext = os.path.splitext(onnx_name)[-1]
    if ext == ".zip":
        onnx_name_unzipped = os.path.join(dest, "large_model", "__MODEL_PROTO.onnx")
        if not os.path.exists(onnx_name_unzipped):
            if verbose:
                print("Unzip model in %r." % os.path.join(dest, "large_model"))
            with zipfile.ZipFile(onnx_name, 'r') as z:
              z.extractall(os.path.join(dest, "large_model"))
        onnx_name = onnx_name_unzipped

    # tflite
    if convert_tflite and not os.path.exists(convert_tflite):
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(tname)
        print('TFL-i:', converter.inference_input_type)
        print('TFL-o:', converter.inference_output_type)
        tflite_model = converter.convert()
        with open(convert_tflite, 'wb') as f:
            f.write(tflite_model)

    # Benchmarks both models.
    if optimize:
        ort = onnxruntime.InferenceSession(onnx_name)
    else:
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        ort = onnxruntime.InferenceSession(onnx_name)

    if verbose:
        print("ONNX inputs:")
        for a in ort.get_inputs():
            print("  {}: {}, {}".format(a.name, a.type, a.shape))
        print("ONNX outputs:")
        for a in ort.get_outputs():
            print("  {}: {}, {}".format(a.name, a.type, a.shape))

    # onnxruntime
    if output_name is None or ort_name is None:
        index = 0
    else:
        output_names = [o.name for o in ort.get_outputs()]
        if output_name in output_names:
            index = output_names.index(output_name)
        elif ort_name in output_names:
            index = output_names.index(ort_name)
        else:
            index = 0
    if isinstance(imgs[0], dict):
        fct_ort = lambda img: ort.run(None, img)[index]
        fct_orts = lambda img: ort.run(None, img)
    else:
        input_name = ort.get_inputs()[0].name
        fct_ort = lambda img: ort.run(None, {input_name: img})[index]
        fct_orts = lambda img: ort.run(None, {input_name: img})
    results_ort, duration_ort = measure_time(fct_ort, imgs)
    if verbose:
        print("ORT", len(imgs), duration_ort)

    # tensorflow
    if custom_tf is None:
        import tensorflow_hub as hub
        from tensorflow import convert_to_tensor
        if isinstance(imgs[0], OrderedDict):
            imgs_tf = [
                OrderedDict((k, convert_to_tensor(v)) for k, v in img.items())
                for img in imgs]
        else:
            imgs_tf = [convert_to_tensor(img) for img in imgs]
        model = hub.load(url.split("?")[0])
        if signature is not None:
            model = model.signatures[signature]
        results_tf, duration_tf = measure_time(model, imgs_tf)
    else:
        output, results_tf, duration_tf = custom_tf(tname)

    if verbose:
        print("TF", len(imgs), duration_tf)
        mean_ort = sum(duration_ort) / len(duration_ort)
        mean_tf = sum(duration_tf) / len(duration_tf)
        print("ratio ORT=%r / TF=%r = %r" % (mean_ort, mean_tf, mean_ort / mean_tf))

    # checks discrepencies
    if custom_tf is None:
        res = model(imgs_tf[0])
    else:
        res = output
    if isinstance(res, dict):
        if output_name is None:
            if len(res) != 1:
                raise NotImplementedError(
                    "TF output contains more than one output=%r and output names=%r." % (
                        list(res), [o.name for o in ort.get_outputs()]))
            else:
                output_name = ort.get_outputs()[0].name
        if output_name not in res:
            raise AssertionError("Unable to find output %r in %r." % (output_name, list(sorted(res))))
        res = res[output_name]
    res_ort = fct_orts(imgs[0])
    try:
        if len(res_ort) > 1:
            check_discrepencies(res_ort, res, threshold)
        else:
            check_discrepencies(res_ort, res.numpy(), threshold)
    except AttributeError as e:
        raise AssertionError(
            "Unable to check discrepencies for res=%r." % res) from e
    except AssertionError as e:
        output_names = [o.name for o in ort.get_outputs()]
        res = fct_ort(imgs[0])
        for i, r in enumerate(res):
            print("ORT %d: %s: %r: %r" % (i, output_names[i], r.dtype, r.shape))
        raise e
    return duration_ort, duration_tf


def benchmark_tflite(url, dest, onnx_name, opset, imgs, verbose=True, threshold=1e-3,
                     names=None):
    """
    Runs a simple benchmark with a tflite model.
    Goes through every steps (download, convert).
    Skips them if already done.
    """
    if url.startswith('http'):
        tname = download_tflite(url, dest)
        if verbose:
            print("Created %r." % tname)
    else:
        tname = url

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

    # tflite
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(tname)
    #help(interpreter)
    input_details = interpreter.get_input_details()
    index_in = input_details[0]['index']
    output_details = interpreter.get_output_details()
    index_out = output_details[0]['index']
    interpreter.allocate_tensors()

    def call_tflite(inp):
        interpreter.set_tensor(index_in, inp)
        interpreter.invoke()
        scores = interpreter.get_tensor(index_out)
        return scores

    # check intermediate results
    if names is not None:
        from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
        import onnx

        with open(onnx_name, "rb") as f:
            model_onnx = onnx.load(f)

        interpreter_details = tf.lite.Interpreter(tname, experimental_preserve_all_tensors=True)
        input_details = interpreter_details.get_input_details()
        index_in = input_details[0]['index']
        interpreter_details.allocate_tensors()
        interpreter_details.set_tensor(index_in, imgs[0])
        interpreter_details.invoke()
        details = interpreter_details.get_tensor_details()

        inputs = {input_name: imgs[0]}
        names_index = {}
        for tt in details:
            names_index[tt['name']] = (tt['index'], tt['quantization'], tt['quantization_parameters'])

        num_results = []
        for name_tfl, name_ort in names:
            index = names_index[name_tfl]
        
            tfl_value = interpreter_details.get_tensor(index[0])
            
            new_name = onnx_name + ".%s.onnx" % name_ort.replace(":", "_").replace(";", "_").replace("/", "_")
            if not os.path.exists(new_name):
                print('[create onnx model for %r, %r.' % (name_tfl, name_ort))
                new_model = select_model_inputs_outputs(model_onnx, outputs=[name_ort])            
                with open(new_name, "wb") as f:
                    f.write(new_model.SerializeToString())

            ort_inter = onnxruntime.InferenceSession(new_name)
            result = ort_inter.run(None, inputs)[0]
            
            diff = numpy.abs(tfl_value.ravel().astype(numpy.float64) -
                             result.ravel().astype(numpy.float64)).max()
            num_results.append("diff=%f names=(%r,%r) " % (diff, name_tfl, name_ort))
            print("*** diff=%f names=(%r,%r) " % (diff, name_tfl, name_ort))
            print("    TFL:", tfl_value.dtype, tfl_value.shape, tfl_value.min(), tfl_value.max()) 
            print("    ORT:", result.dtype, result.shape, result.min(), result.max()) 
        
        print("\n".join(num_results))

    results_tfl, duration_tfl = measure_time(call_tflite, imgs)

    if verbose:
        print("TFL", len(imgs), duration_tfl)
        mean_ort = sum(duration_ort) / len(duration_ort)
        mean_tfl = sum(duration_tfl) / len(duration_tfl)
        print("ratio ORT=%r / TF=%r = %r" % (mean_ort, mean_tfl, mean_ort / mean_tfl))
    
    # checks discrepencies
    res = call_tflite(imgs[0])
    res_ort = fct_ort(imgs[0])
    if isinstance(res, dict):
        if len(res) != 1:
            raise NotImplementedError("TF output contains more than one output: %r." % res)
        output_name = ort.get_outputs()[0].name
        if output_name not in res:
            raise AssertionError("Unable to find output %r in %r." % (output_name, list(sorted(res))))
        res = res[output_name]
        
    check_discrepencies(res_ort, res, threshold)
    return duration_ort, duration_tf
