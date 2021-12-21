# SPDX-License-Identifier: Apache-2.0

import os
from os.path import dirname, abspath
import unittest
import sys
sys.path.insert(0, os.path.join(dirname(abspath(__file__)), '../../keras2onnx_tests/'))

import json
import urllib.request
import pickle
import numpy as np
import tensorflow as tf

from onnxconverter_common.onnx_ex import get_maximum_opset_supported
import mock_keras2onnx
from mock_keras2onnx.proto import keras, is_tensorflow_older_than
from test_utils import is_bloburl_access, run_onnx_runtime

enable_full_transformer_test = False
if os.environ.get('ENABLE_FULL_TRANSFORMER_TEST', '0') != '0':
    enable_transformer_test = True

CONVERTER_TRANSFERMER_PATH = r'https://lotus.blob.core.windows.net/converter-models/transformer_tokenizer/'


@unittest.skipIf(is_tensorflow_older_than('2.1.0'),
                 "Transformers conversion need tensorflow 2.1.0+")
@unittest.skipIf(not is_bloburl_access(CONVERTER_TRANSFERMER_PATH), "Model blob url can't access.")
class TestTransformers(unittest.TestCase):

    text_str = 'The quick brown fox jumps over lazy dog.'

    def setUp(self):
        self.model_files = []

    def tearDown(self):
        for fl in self.model_files:
            os.remove(fl)

    def _get_token_path(self, file_name):
        return 'https://lotus.blob.core.windows.net/converter-models/transformer_tokenizer/' + file_name

    def _get_tokenzier(self, tokenizer_file):
        token_path = self._get_token_path(tokenizer_file)
        if not os.path.exists(tokenizer_file):
            urllib.request.urlretrieve(token_path, tokenizer_file)
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    def _prepare_inputs(self, tokenizer, batch_size=3):
        raw_data = json.dumps({
            'text': self.text_str
        })
        text = json.loads(raw_data)['text']
        # The tokenizers are generated using transformers 2.5.0, but model_max_length is introduced and needed in 2.9.0.
        if not hasattr(tokenizer, 'model_max_length'):
            tokenizer.model_max_length = 1024
        inputs_raw = tokenizer.encode_plus(text, add_special_tokens=True)
        idx_not_None = [i_ for i_, v_ in enumerate(inputs_raw.data['input_ids']) if v_ is not None]
        input_raw_not_None = inputs_raw if len(idx_not_None) == len(inputs_raw.data['input_ids']) else \
            {k_: [v_[i_] for i_ in idx_not_None] for k_, v_ in inputs_raw.items()}
        inputs_onnx = {k_: np.repeat(np.expand_dims(v_, axis=0), batch_size, axis=0) for k_, v_ in input_raw_not_None.items()}
        inputs = {k_: tf.constant(v_) for k_, v_ in inputs_onnx.items()}
        return text, inputs, inputs_onnx

    @unittest.skip("Output shape mismatch for tf model prediction.")
    def test_3layer_gpt2(self):
        from transformers import GPT2Config, TFGPT2Model, BertTokenizer
        mock_keras2onnx.proto.keras.backend.set_learning_phase(0)
        config = GPT2Config(n_layer=3)
        model = TFGPT2Model(config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='tf')
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertModel(self):
        from transformers import BertConfig, TFBertModel
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertModel(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFBertForPreTraining(self):
        from transformers import BertConfig, TFBertForPreTraining
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForPreTraining(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFBertForMaskedLM(self):
        from transformers import BertConfig, TFBertForMaskedLM
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForMaskedLM(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFBertForNextSentencePrediction(self):
        from transformers import BertConfig, TFBertForNextSentencePrediction
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForNextSentencePrediction(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForSequenceClassification(self):
        from transformers import BertConfig, TFBertForSequenceClassification
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForSequenceClassification(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForTokenClassification(self):
        from transformers import BertConfig, TFBertForTokenClassification
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForTokenClassification(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFBertForQuestionAnswering(self):
        from transformers import BertConfig, TFBertForQuestionAnswering
        keras.backend.clear_session()
        # pretrained_weights = 'bert-base-uncased'
        tokenizer_file = 'bert_bert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = BertConfig()
        model = TFBertForQuestionAnswering(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFGPT2(self):
        if enable_full_transformer_test:
            from transformers import GPT2Config, TFGPT2Model, TFGPT2LMHeadModel, TFGPT2DoubleHeadsModel
            model_list = [TFGPT2Model, TFGPT2LMHeadModel, TFGPT2DoubleHeadsModel]
        else:
            from transformers import GPT2Config, TFGPT2Model
            model_list = [TFGPT2Model]
        # pretrained_weights = 'gpt2'
        tokenizer_file = 'gpt2_gpt2.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = GPT2Config()
        for model_instance_ in model_list:
            keras.backend.clear_session()
            model = model_instance_(config)
            model._set_inputs(inputs)
            predictions_original = model(inputs)
            predictions = [predictions_original[0]] + list(v_.numpy() for v_ in predictions_original[1])
            onnx_model = mock_keras2onnx.convert_keras(model, model.name)
            self.assertTrue(
                run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                                 atol=1.e-4))

    @unittest.skipIf(get_maximum_opset_supported() < 12, "Einsum is not supported until opset 12.")
    def test_TFXLNet(self):
        if enable_full_transformer_test:
            from transformers import XLNetConfig, TFXLNetModel, TFXLNetLMHeadModel, TFXLNetForSequenceClassification, \
                TFXLNetForTokenClassification, TFXLNetForQuestionAnsweringSimple, XLNetTokenizer
            model_list = [TFXLNetModel, TFXLNetLMHeadModel, TFXLNetForSequenceClassification, \
                TFXLNetForTokenClassification, TFXLNetForQuestionAnsweringSimple]
        else:
            from transformers import XLNetConfig, TFXLNetModel, XLNetTokenizer
            model_list = [TFXLNetModel]

        # XLNetTokenizer need SentencePiece, so the pickle file does not work here.
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        config = XLNetConfig(n_layer=2)
        # The model with input mask has MatrixDiagV3 which is not a registered function/op
        token = np.asarray(tokenizer.encode(self.text_str, add_special_tokens=True), dtype=np.int32)
        inputs_onnx = {'input_1': np.expand_dims(token, axis=0)}
        inputs = tf.constant(token)[None, :]  # Batch size 1

        for model_instance_ in model_list:
            keras.backend.clear_session()
            model = model_instance_(config)
            predictions = model.predict(inputs)
            onnx_model = mock_keras2onnx.convert_keras(model)
            self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFOpenAIGPTModel(self):
        from transformers import OpenAIGPTConfig, TFOpenAIGPTModel
        keras.backend.clear_session()
        # pretrained_weights = 'openai-gpt'
        tokenizer_file = 'openai_openai-gpt.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = OpenAIGPTConfig()
        model = TFOpenAIGPTModel(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFOpenAIGPTLMHeadModel(self):
        from transformers import OpenAIGPTConfig, TFOpenAIGPTLMHeadModel
        keras.backend.clear_session()
        # pretrained_weights = 'openai-gpt'
        tokenizer_file = 'openai_openai-gpt.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = OpenAIGPTConfig()
        model = TFOpenAIGPTLMHeadModel(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFOpenAIGPTDoubleHeadsModel(self):
        from transformers import OpenAIGPTConfig, TFOpenAIGPTDoubleHeadsModel
        keras.backend.clear_session()
        # pretrained_weights = 'openai-gpt'
        tokenizer_file = 'openai_openai-gpt.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        # tf.gather(hidden_states, cls_index, batch_dims=len(hidden_shape) - 2), batch_dims = 1 in this case
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer, batch_size=1)
        config = OpenAIGPTConfig()
        model = TFOpenAIGPTDoubleHeadsModel(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skip('tensorflow.GraphDef exceeds maximum protobuf size of 2GB')
    def test_TFXLMModel(self):
        from transformers import XLMConfig, TFXLMModel
        keras.backend.clear_session()
        # pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer_file = 'xlm_xlm-mlm-enfr-1024.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = XLMConfig()
        model = TFXLMModel(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skip('tensorflow.GraphDef exceeds maximum protobuf size of 2GB')
    def test_TFXLMWithLMHeadModel(self):
        from transformers import XLMConfig, TFXLMWithLMHeadModel
        keras.backend.clear_session()
        # pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer_file = 'xlm_xlm-mlm-enfr-1024.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = XLMConfig()
        model = TFXLMWithLMHeadModel(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skip('tensorflow.GraphDef exceeds maximum protobuf size of 2GB')
    def test_TFXLMForSequenceClassification(self):
        from transformers import XLMConfig, TFXLMForSequenceClassification
        keras.backend.clear_session()
        # pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer_file = 'xlm_xlm-mlm-enfr-1024.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = XLMConfig()
        model = TFXLMForSequenceClassification(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    @unittest.skip('tensorflow.GraphDef exceeds maximum protobuf size of 2GB')
    def test_TFXLMForQuestionAnsweringSimple(self):
        from transformers import XLMConfig, TFXLMForQuestionAnsweringSimple
        keras.backend.clear_session()
        # pretrained_weights = 'xlm-mlm-enfr-1024'
        tokenizer_file = 'xlm_xlm-mlm-enfr-1024.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = XLMConfig()
        model = TFXLMForQuestionAnsweringSimple(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertModel(self):
        from transformers import DistilBertConfig, TFDistilBertModel
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertModel(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForMaskedLM(self):
        from transformers import DistilBertConfig, TFDistilBertForMaskedLM
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertForMaskedLM(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFDistilBertForSequenceClassification(self):
        from transformers import DistilBertConfig, TFDistilBertForSequenceClassification
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertForSequenceClassification(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForTokenClassification(self):
        from transformers import DistilBertConfig, TFDistilBertForTokenClassification
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertForTokenClassification(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFDistilBertForQuestionAnswering(self):
        from transformers import DistilBertConfig, TFDistilBertForQuestionAnswering
        keras.backend.clear_session()
        # pretrained_weights = 'distilbert-base-uncased'
        tokenizer_file = 'distilbert_distilbert-base-uncased.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = DistilBertConfig()
        model = TFDistilBertForQuestionAnswering(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFRobertaModel(self):
        from transformers import RobertaConfig, TFRobertaModel
        keras.backend.clear_session()
        # pretrained_weights = 'roberta-base'
        tokenizer_file = 'roberta_roberta-base.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = RobertaConfig()
        model = TFRobertaModel(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    def test_TFRobertaForMaskedLM(self):
        from transformers import RobertaConfig, TFRobertaForMaskedLM
        keras.backend.clear_session()
        # pretrained_weights = 'roberta-base'
        tokenizer_file = 'roberta_roberta-base.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = RobertaConfig()
        model = TFRobertaForMaskedLM(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(
            run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files, rtol=1.e-2,
                             atol=1.e-4))

    def test_TFRobertaForSequenceClassification(self):
        from transformers import RobertaConfig, TFRobertaForSequenceClassification
        keras.backend.clear_session()
        # pretrained_weights = 'roberta-base'
        tokenizer_file = 'roberta_roberta-base.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = RobertaConfig()
        model = TFRobertaForSequenceClassification(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))

    @unittest.skipIf(not enable_full_transformer_test, "Full transfomer test is not enabled")
    def test_TFRobertaForTokenClassification(self):
        from transformers import RobertaConfig, TFRobertaForTokenClassification
        keras.backend.clear_session()
        # pretrained_weights = 'roberta-base'
        tokenizer_file = 'roberta_roberta-base.pickle'
        tokenizer = self._get_tokenzier(tokenizer_file)
        text, inputs, inputs_onnx = self._prepare_inputs(tokenizer)
        config = RobertaConfig()
        model = TFRobertaForTokenClassification(config)
        predictions = model.predict(inputs)
        onnx_model = mock_keras2onnx.convert_keras(model, model.name)
        self.assertTrue(run_onnx_runtime(onnx_model.graph.name, onnx_model, inputs_onnx, predictions, self.model_files))


if __name__ == "__main__":
    unittest.main()
