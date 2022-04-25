# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TF 2.0 RoBERTa model. """


import logging

import tensorflow as tf

from xmixup.models.bert import BertEmbeddings, BertMainLayer
from xmixup.utils.model_utils import shape_list, keras_serializable
from xmixup.models.pretrain_model import RobertaPreTrainedModel

logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "RobertaTokenizer"


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.padding_idx = 1

    def create_position_ids_from_input_ids(self, x):
        """ Replace non-padding symbols with their position numbers. Position numbers begin at
        padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
        `utils.make_positions`.
        :param tf.Tensor x:
        :return tf.Tensor:
        """
        mask = tf.cast(tf.math.not_equal(x, self.padding_idx), dtype=tf.int32)
        incremental_indicies = tf.math.cumsum(mask, axis=1) * mask
        return incremental_indicies + self.padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.
        :param tf.Tensor inputs_embeds:
        :return tf.Tensor:
        """
        seq_length = shape_list(inputs_embeds)[1]

        position_ids = tf.range(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=tf.int32)[tf.newaxis, :]
        return position_ids

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super()._embedding([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)


@keras_serializable
class RobertaMainLayer(BertMainLayer):
    """
    Same as TFBertMainLayer but uses TFRobertaEmbeddings.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.embeddings = RobertaEmbeddings(config, name="embeddings")


class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.roberta = RobertaMainLayer(config, name="roberta")

    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`tf.Tensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        outputs = self.roberta(inputs, **kwargs)
        return outputs
