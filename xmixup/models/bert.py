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
""" TF 2.0 BERT model. """


import logging

import tensorflow as tf

from xmixup.configuration.xlmr import BertConfig
from xmixup.utils.model_utils import keras_serializable, get_initializer, shape_list, cast_bool_to_primitive
from xmixup.utils.activations import gelu
from xmixup.tokenization.utils import BatchEncoding
from xmixup.models.pretrain_model import PreTrainedModel


logger = logging.getLogger(__name__)

_TOKENIZER_FOR_DOC = "BertTokenizer"
do_cast_bool_to_primitive = False

def swish(x):
    return x * tf.sigmoid(x)


ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.layers.Activation(swish),
}


class BertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range

        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="position_embeddings",
        )
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=get_initializer(self.initializer_range),
            name="token_type_embeddings",
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope("word_embeddings"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.word_embeddings = self.add_weight(
                "weight",
                shape=[self.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )
        super().build(input_shape)

    def call(self, inputs, mode="embedding", training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == "embedding":
            return self._embedding(inputs, training=training)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs, training=False):
        """Applies embedding based on inputs tensor."""
        input_ids, position_ids, token_type_ids, inputs_embeds = inputs

        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
            Args:
                inputs: A float32 tensor with shape [batch_size, length, hidden_size]
            Returns:
                float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])


class BertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False, cross_attention=False):
        if not cross_attention:
            hidden_states, attention_mask, head_mask, output_attentions = inputs
        else:
            hidden_states, attention_mask, hidden_states_trg, attention_mask_trg, head_mask, output_attentions = inputs

        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        if not cross_attention:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states_trg)
            mixed_value_layer = self.value(hidden_states_trg)
            attention_mask = attention_mask_trg

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(
            query_layer, key_layer, transpose_b=True
        )  # (batch size, num_heads, seq_len_q, seq_len_k)
        dk = tf.cast(shape_list(key_layer)[-1], tf.float32)  # scale attention_scores
        attention_scores = attention_scores / tf.math.sqrt(dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            masked_attention_scores = attention_scores + attention_mask
        else:
            masked_attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(masked_attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, -1, self.all_head_size)
        )  # (batch_size, seq_len_q, all_head_size)

        outputs = (
            (context_layer, attention_scores) #if (cast_bool_to_primitive(output_attentions) or not do_cast_bool_to_primitive) else (context_layer,)
        )

        return outputs


class BertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False, ratio=None):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        if ratio is None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(tf.multiply(ratio, hidden_states) + tf.multiply(1.0 - ratio, input_tensor))
        return hidden_states


class BertAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = BertSelfAttention(config, name="self")
        self.dense_output = BertSelfOutput(config, name="output")

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, inputs, training=False, cross_attention=False, ratio=None):
        if not cross_attention:
            input_tensor, attention_mask, head_mask, output_attentions = inputs
        else:
            input_tensor, attention_mask, input_tensor_trg, attention_mask_trg, head_mask, output_attentions = inputs

        self_outputs = self.self_attention(
            inputs, training=training, cross_attention=cross_attention
        )
        attention_output = self.dense_output([self_outputs[0], input_tensor], training=training, ratio=ratio)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = BertAttention(config, name="attention")
        self.intermediate = BertIntermediate(config, name="intermediate")
        self.bert_output = BertOutput(config, name="output")

    def call(self, inputs, training=False, add_attention=True, add_ffn=True, cross_attention=False, ratio=None):
        if add_attention:
            attention_outputs = self.attention(
                inputs, training=training, cross_attention=cross_attention, ratio=ratio
            )
            if not add_ffn:
                return attention_outputs
            attention_output = attention_outputs[0]
        else:
            attention_output = inputs[0]
            attention_outputs = (attention_output, None)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output([intermediate_output, attention_output], training=training)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.layer = [BertLayer(config, name="layer_._{}".format(i)) for i in range(self.num_hidden_layers)]

    def call(self, inputs, training=False, start_layer=0, end_layer=None, add_attention=True, add_ffn=True,
             cross_attention=False, ratio=None):
        if end_layer is None:
            end_layer = self.num_hidden_layers
        if not cross_attention:
            hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states = inputs
        else:
            hidden_states, attention_mask, hidden_states_trg, attention_mask_trg, head_mask, output_attentions, output_hidden_states = inputs

        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer[start_layer: end_layer]):
            if cast_bool_to_primitive(output_hidden_states) is True or not do_cast_bool_to_primitive:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if not cross_attention:
                layer_outputs = layer_module(
                    [hidden_states, attention_mask, head_mask[i], output_attentions], training=training,
                    add_attention=add_attention, add_ffn=add_ffn, cross_attention=cross_attention
                )
            else:
                layer_outputs = layer_module(
                    [hidden_states, attention_mask, hidden_states_trg, attention_mask_trg, head_mask[i],
                     output_attentions], training=training,
                    add_attention=add_attention, add_ffn=add_ffn, cross_attention=cross_attention, ratio=ratio
                )
            hidden_states = layer_outputs[0]

            if cast_bool_to_primitive(output_attentions) is True or not do_cast_bool_to_primitive:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if cast_bool_to_primitive(output_hidden_states) is True or not do_cast_bool_to_primitive:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if cast_bool_to_primitive(output_hidden_states) is True or not do_cast_bool_to_primitive:
            outputs = outputs + (all_hidden_states,)
        if cast_bool_to_primitive(output_attentions) is True or not do_cast_bool_to_primitive:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


class BertPooler(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


@keras_serializable
class BertMainLayer(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.embeddings = BertEmbeddings(config, name="embeddings")
        self.encoder = BertEncoder(config, name="encoder")
        self.pooler = BertPooler(config, name="pooler")

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        raise NotImplementedError

    def call(
        self,
        inputs,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        training=False,
    ):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
            position_ids = inputs[3] if len(inputs) > 3 else position_ids
            head_mask = inputs[4] if len(inputs) > 4 else head_mask
            inputs_embeds = inputs[5] if len(inputs) > 5 else inputs_embeds
            output_attentions = inputs[6] if len(inputs) > 6 else output_attentions
            output_hidden_states = inputs[7] if len(inputs) > 7 else output_hidden_states
            assert len(inputs) <= 8, "Too many inputs."
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
            position_ids = inputs.get("position_ids", position_ids)
            head_mask = inputs.get("head_mask", head_mask)
            inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
            output_attentions = inputs.get("output_attentions", output_attentions)
            output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
            assert len(inputs) <= 8, "Too many inputs."
        else:
            input_ids = inputs

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        embedding_output = self.embeddings([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)
        encoder_outputs = self.encoder(
            [embedding_output, extended_attention_mask, head_mask, output_attentions, output_hidden_states],
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"


class BertModel(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertMainLayer(config, name="bert")

    def call(self, inputs, **kwargs):
        r"""
    Returns:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
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
        outputs = self.bert(inputs, **kwargs)
        return outputs
