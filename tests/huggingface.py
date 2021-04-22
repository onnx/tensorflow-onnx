# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for huggingface tensorflow transformers.

tested with tf-2.4.1, transformers-4.5.1

"""

# pylint: disable=missing-docstring,invalid-name,unused-argument
# pylint: disable=bad-classmethod-argument,wrong-import-position
# pylint: disable=import-outside-toplevel

import os
import time
import unittest
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tf2onnx

compare_perf = True
time_to_run = 10
time_step = 10


class TestTransformers(unittest.TestCase):

    def setUp(self):
        tf.compat.v1.reset_default_graph()

    @classmethod
    def assertAllClose(cls, expected, actual, **kwargs):
        np.testing.assert_allclose(expected, actual, **kwargs)

    def run_onnxruntime(self, model_path, input_dict, output_names):
        """Run test against onnxruntime backend."""
        providers = ['CPUExecutionProvider']
        if rt.get_device() == "GPU":
            gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
            if gpus is None or len(gpus) > 1:
                providers = ['CUDAExecutionProvider']

        opt = rt.SessionOptions()
        # in case of issues with the runtime, one can enable more logging
        # opt.log_severity_level = 0
        # opt.log_verbosity_level = 255
        # opt.enable_profiling = True
        m = rt.InferenceSession(model_path, sess_options=opt, providers=providers)
        results = m.run(output_names, input_dict)
        if compare_perf:
            n = 0
            time_start = time.time()
            time_stop = time_start + time_to_run
            while time.time() < time_stop:
                for _ in range(time_step):
                    _ = m.run(output_names, input_dict)
                n += time_step
            time_end = time.time()
            val = (time_end - time_start) / n
            print(f'= avg ort name={self.name}, time={val}, n={n}')
        return results

    def run_keras(self, model, inputs):
        pred = model(inputs)
        if compare_perf:
            n = 0
            time_start = time.time()
            time_stop = time_start + time_to_run
            while time.time() < time_stop:
                for _ in range(time_step):
                    _ = model(inputs)
                n += time_step
            time_stop = time.time()
            val = (time_stop - time_start) / n
            print(f'= avg keras name={self.name}, time={val}, n={n}')
        return pred

    def run_test(self, model, input_dict, rtol=1e-2, atol=1e-4, input_signature=None,
                 outputs=None, large=True, extra_input=None):

        # always use external model format for consistency
        large = True
        self.name = self._testMethodName.replace("test_", "")
        print(f"==== {self.name}")
        dst = os.path.join("/tmp", "test_transformers", self.name)
        os.makedirs(dst, exist_ok=True)

        # run keras model
        print("= running keras")
        tf_results = self.run_keras(model, input_dict)
        if not outputs:
            # no outputs given ... take all
            outputs = list(tf_results.keys())

        # filter outputs
        tf_results = [v.numpy() for k, v in tf_results.items() if k in outputs]

        # input tensors to numpy
        input_dict = {k: v.numpy() for k, v in input_dict.items()}

        model_path = os.path.join(dst, self.name)
        if not large:
            model_path = model_path + ".onnx"
        print("= convert")
        time_start = time.time()
        _, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature,
                                          opset=13, large_model=large, output_path=model_path)
        time_stop = time.time()
        print(f"= convertsion took {time_stop - time_start}")

        if large:
            # need to unpack the zip for run_onnxruntime()
            with zipfile.ZipFile(model_path, 'r') as z:
                z.extractall(os.path.dirname(model_path))
            model_path = os.path.join(os.path.dirname(model_path), "__MODEL_PROTO.onnx")

        print("= running ort")
        if extra_input:
            input_dict.update(extra_input)
        onnx_results = self.run_onnxruntime(model_path, input_dict, outputs)
        self.assertAllClose(tf_results, onnx_results, rtol=rtol, atol=atol)

    def spec_and_pad(self, input_dict, max_length=None, batchdim=None):
        spec = []
        new_dict = {}
        for k, v in input_dict.items():
            shape = v.shape
            if len(shape) == 2:
                if not max_length:
                    shape = [batchdim, None]
                else:
                    shape = [batchdim, max_length]
            spec.append(tf.TensorSpec(shape, dtype=v.dtype, name=k))
            if max_length:
                l = len(v[0])
                v = tf.pad(v, [[0, 0], [0, max_length-l]])
            new_dict[k] = v
        return tuple(spec), new_dict

    # BERT

    def test_TFBertModel(self):
        from transformers import BertTokenizer, TFBertForQuestionAnswering
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = TFBertForQuestionAnswering.from_pretrained('bert-base-cased')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_dict = tokenizer(question, text, return_tensors='tf')
        spec, input_dict = self.spec_and_pad(input_dict)
        self.run_test(model, input_dict, input_signature=spec)

    def test_TFBertFineTunedSquadModel(self):
        from transformers import BertTokenizer, TFBertForQuestionAnswering
        name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        tokenizer = BertTokenizer.from_pretrained(name)
        model = TFBertForQuestionAnswering.from_pretrained(name)
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_dict = tokenizer(question, text, return_tensors='tf')
        spec, input_dict = self.spec_and_pad(input_dict)
        self.run_test(model, input_dict, input_signature=spec)

    def test_TFDisillBertModel(self):
        from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
        name = 'distilbert-base-uncased'
        name = 'distilbert-base-uncased-distilled-squad'
        tokenizer = DistilBertTokenizer.from_pretrained(name)
        model = TFDistilBertForQuestionAnswering.from_pretrained(name)
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_dict = tokenizer(question, text, return_tensors='tf')
        spec, input_dict = self.spec_and_pad(input_dict)
        outputs = ["start_logits", "end_logits"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, rtol=1e-5)

    ## FUNNEL

    def _test_TFFunnel(self, size, large=False):
        from transformers import FunnelTokenizer, TFFunnelForQuestionAnswering
        tokenizer = FunnelTokenizer.from_pretrained(size)
        model = TFFunnelForQuestionAnswering.from_pretrained(size)
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_dict = tokenizer(question, text, return_tensors='tf')
        spec, input_dict = self.spec_and_pad(input_dict, 128)
        outputs = ["start_logits", "end_logits"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, rtol=1e-5)

    def test_TFFunnelSmall(self):
        self._test_TFFunnel("funnel-transformer/small")

    def test_TFFunnelSmallBase(self):
        self._test_TFFunnel("funnel-transformer/small-base")

    def test_TFFunnelMedium(self):
        self._test_TFFunnel("funnel-transformer/medium")

    def test_TFFunnelMediumBase(self):
        self._test_TFFunnel("funnel-transformer/medium-base")

    def test_TFFunnelIntermediate(self):
        self._test_TFFunnel("funnel-transformer/intermediate")

    def test_TFFunnelIntermediateBase(self):
        self._test_TFFunnel("funnel-transformer/intermediate-base")

    def test_TFFunnelLarge(self):
        self._test_TFFunnel("funnel-transformer/large")

    def test_TFFunnelLargeBase(self):
        self._test_TFFunnel("funnel-transformer/large-base")

    def test_TFFunnelXLarge(self):
        self._test_TFFunnel("funnel-transformer/xlarge")

    def test_TFFunnelXLargeBase(self):
        self._test_TFFunnel("funnel-transformer/xlarge-base")

    ## T5

    def _test_TFT5Model(self, size, large=False):
        from transformers import T5Tokenizer, TFT5Model
        tokenizer = T5Tokenizer.from_pretrained(size)
        model = TFT5Model.from_pretrained(size)
        input_ids = \
            tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="tf").input_ids
        decoder_input_ids = \
            tokenizer("Studies show that", return_tensors="tf").input_ids
        input_dict = {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}
        spec, input_dict = self.spec_and_pad(input_dict)
        outputs = ["last_hidden_state"]
        self.run_test(model, input_dict,
                      input_signature=spec, outputs=outputs, large=large)

    def test_TFT5ModelSmall(self):
        self._test_TFT5Model("t5-small")

    def test_TFT5ModelBase(self):
        self._test_TFT5Model("t5-base")

    def test_TFT5ModelLarge(self):
        self._test_TFT5Model("t5-large", large=True)

    def test_TFT5Model3B(self):
        self._test_TFT5Model("t5-3b", large=True)

    def test_TFT5Model11B(self):
        self._test_TFT5Model("t5-11b", large=True)

    ## Albert

    def _test_TFAlbert(self, size, large=False):
        from transformers import AlbertTokenizer, TFAlbertModel
        tokenizer = AlbertTokenizer.from_pretrained(size)
        model = TFAlbertModel.from_pretrained(size)
        input_dict = tokenizer("Hello, my dog is cute", return_tensors="tf")
        spec, input_dict = self.spec_and_pad(input_dict)
        outputs = ["last_hidden_state"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, large=large)

    def test_TFAlbertBaseV1(self):
        self._test_TFAlbert("albert-base-v1", large=True)

    def test_TFAlbertLargeV1(self):
        self._test_TFAlbert("albert-large-v1", large=True)

    def test_TFAlbertXLargeV1(self):
        self._test_TFAlbert("albert-xlarge-v1", large=True)

    def test_TFAlbertXXLargeV1(self):
        self._test_TFAlbert("albert-xxlarge-v1", large=True)

    def test_TFAlbertBaseV2(self):
        self._test_TFAlbert("albert-base-v2")

    def test_TFAlbertLargeV2(self):
        self._test_TFAlbert("albert-large-v2", large=True)

    def test_TFAlbertXLargeV2(self):
        self._test_TFAlbert("albert-xlarge-v2", large=True)

    def test_TFAlbertXXLargeV2(self):
        self._test_TFAlbert("albert-xxlarge-v2", large=True)

    # CTRL

    def test_TFCTRL(self):
        from transformers import CTRLTokenizer, TFCTRLModel
        tokenizer = CTRLTokenizer.from_pretrained('ctrl')
        model = TFCTRLModel.from_pretrained('ctrl')
        input_dict = tokenizer("Hello, my dog is cute", return_tensors="tf")
        spec, input_dict = self.spec_and_pad(input_dict)
        outputs = ["last_hidden_state"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, large=True)

    def _test_TFGpt2(self, size, large=False):
        from transformers import GPT2Tokenizer, TFGPT2Model
        tokenizer = GPT2Tokenizer.from_pretrained(size)
        model = TFGPT2Model.from_pretrained(size)
        input_dict = tokenizer("Hello, my dog is cute", return_tensors="tf")
        spec, input_dict = self.spec_and_pad(input_dict)
        outputs = ["last_hidden_state"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, large=large)

    # GPT2

    def test_TFGpt2(self):
        self._test_TFGpt2("gpt2")

    def test_TFGpt2Large(self):
        self._test_TFGpt2("gpt2-large", large=True)

    def test_TFGpt2XLarge(self):
        self._test_TFGpt2("gpt2-xl", large=True)

    def test_TFDialoGPT(self):
        self._test_TFGpt2("microsoft/DialoGPT-large", large=True)

    # LONGFORMER

    def _test_TFLongformer(self, size, large=False):
        from transformers import LongformerTokenizer, TFLongformerModel
        tokenizer = LongformerTokenizer.from_pretrained(size)
        model = TFLongformerModel.from_pretrained(size)
        input_dict = tokenizer("Hello, my dog is cute", return_tensors="tf")
        spec, input_dict = self.spec_and_pad(input_dict, max_length=512)
        outputs = ["last_hidden_state"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, large=large)

    def test_TFLongformerBase(self):
        # fails since transformers-2.4.2?
        #
        # transformers/models/longformer/modeling_tf_longformer.py", line 1839, in _pad_to_window_size
        # if tf.math.greater(padding_len, 0)
        # OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed
        #
        self._test_TFLongformer("allenai/longformer-base-4096", large=True)

    def test_TFLongformerLarge(self):
        self._test_TFLongformer("allenai/longformer-large-4096", large=True)

    # PEGASUS

    def _test_TFPegasus(self, size, large=False):
        from transformers import PegasusTokenizer, TFPegasusModel
        tokenizer = PegasusTokenizer.from_pretrained(size)
        model = TFPegasusModel.from_pretrained(size)
        input_ids = \
            tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="tf").input_ids
        decoder_input_ids = \
            tokenizer("Studies show that", return_tensors="tf").input_ids

        input_dict = {"input_ids": input_ids, "decoder_input_ids": decoder_input_ids}

        # this comes from TFPegasusEncoder/Decoder like:
        #   self.embed_scale = tf.math.sqrt(float(config.d_model)) if config.scale_embedding else 1.0
        # while this is mean to come from config tf tells us that those are model inputs
        # this might be new in tensformers-2.4.2, we did not notice before that
        extra_input = {"tf_pegasus_model/model/decoder/mul/y:0": np.array([32.], dtype=np.float32),
                       "tf_pegasus_model/model/encoder/mul/y:0": np.array([32.], dtype=np.float32)}
        spec, input_dict = self.spec_and_pad(input_dict, max_length=model.config.max_length)
        outputs = ["last_hidden_state"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, large=large, extra_input=extra_input)

    def test_TFPegasus(self):
        self._test_TFPegasus("google/pegasus-xsum", large=True)

    # XLM

    def _test_TFXLM(self, size, large=False):
        from transformers import TFXLMModel, XLMTokenizer
        tokenizer = XLMTokenizer.from_pretrained(size)
        model = TFXLMModel.from_pretrained(size)
        input_dict = tokenizer("Hello, my dog is cute", return_tensors="tf")
        spec, input_dict = self.spec_and_pad(input_dict)
        outputs = ["last_hidden_state"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, large=large, atol=0.005)

    def test_TFXLM(self):
        self._test_TFXLM("xlm-mlm-en-2048", large=True)

    def test_TFXLM_ENDE(self):
        self._test_TFXLM("xlm-mlm-ende-1024", large=True)

    def test_TFXLM_CLMENDE(self):
        self._test_TFXLM("xlm-clm-ende-1024", large=True)

    # BART

    def _test_TFBart(self, size, large=False):
        from transformers import BartTokenizer, TFBartModel
        tokenizer = BartTokenizer.from_pretrained(size)
        model = TFBartModel.from_pretrained(size)
        input_dict = tokenizer("Hello, my dog is cute", return_tensors="tf")
        spec, input_dict = self.spec_and_pad(input_dict, max_length=128)
        outputs = ["last_hidden_state"]
        self.run_test(model, input_dict, input_signature=spec, outputs=outputs, large=large)

    def test_TFBartBase(self):
        self._test_TFBart("facebook/bart-base", large=True)

    def test_TFBartLarge(self):
        self._test_TFBart("facebook/bart-large", large=True)

    def test_TFBartLargeCnn(self):
        self._test_TFBart("facebook/bart-large-cnn", large=True)


if __name__ == "__main__":
    unittest.main()
