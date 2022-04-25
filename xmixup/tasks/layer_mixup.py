import tensorflow as tf
from xmixup.utils.model_utils import shape_list


class XMixup(object):
    def __init__(self, config, **kwargs):
        self.n_langs = config.n_langs
        self.entropy_dense = [tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Ones(),
                                                   name=f"entropy_dense_{i}") for i in range(24)]
        self.lambda_act = tf.keras.layers.Activation(tf.nn.sigmoid)

    @staticmethod
    def get_bert_embedding_output(
                                  roberta,
                                  inputs=None,
                                  attention_mask=None,
                                  token_type_ids=None,
                                  position_ids=None,
                                  inputs_embeds=None,
                                  training=False
                                  ):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask", attention_mask)
        if input_ids is not None:
            input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)

        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask_reverse = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = roberta.embeddings([input_ids, position_ids, token_type_ids, inputs_embeds], training=training)

        return embedding_output, extended_attention_mask, extended_attention_mask_reverse

    def manifold_mixup(
        self,
        roberta,
        inputs=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        training=False,
        mixup_inference=False,
        start_layer=1,
        end_layer=20,
        sample_gold=True,
        lambda_0=0.5
    ):

        if not training and not mixup_inference:
            outputs = roberta(
                inputs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                training=training,
            )

        else:
            if not training or sample_gold:
                input_ids_en = inputs.get("input_ids_en", None)
                attention_mask_en = inputs.get("attention_mask_en", None)
                token_type_ids_en = inputs.get("token_type_ids_en", None)
            else:
                input_ids_en = inputs.get("input_ids_back_trans", None)
                attention_mask_en = inputs.get("attention_mask_back_trans", None)
                token_type_ids_en = inputs.get("token_type_ids_back_trans", None)
            if input_ids_en is not None:
                if token_type_ids_en is None:
                    inputs_en = {"input_ids": input_ids_en, "attention_mask": attention_mask_en}
                else:
                    inputs_en = {"input_ids": input_ids_en, "attention_mask": attention_mask_en,
                                 "token_type_ids": token_type_ids_en}
            else:
                inputs_en = inputs

            if head_mask is not None:
                raise NotImplementedError
            else:
                head_mask = [None] * roberta.num_hidden_layers

            embedding_output, attention_mask, extended_attention_mask = \
                self.get_bert_embedding_output(roberta, inputs, training=training)
            embedding_output_en, attention_mask_en, extended_attention_mask_en = \
                self.get_bert_embedding_output(roberta, inputs_en, training=training)
            lower_encoder_outputs = roberta.encoder(
                [embedding_output, extended_attention_mask, head_mask, output_attentions, output_hidden_states],
                training=training, end_layer=start_layer
            )
            lower_encoder_outputs_en = roberta.encoder(
                [embedding_output_en, extended_attention_mask_en, head_mask, output_attentions, output_hidden_states],
                training=training, end_layer=start_layer
            )
            middle_hidden_states = lower_encoder_outputs[0]  # [batch_size, max_seq_len, hidden_dims]
            middle_hidden_states_en = lower_encoder_outputs_en[0]  # [batch_size, max_seq_len, hidden_dims]

            # manifold mixup
            for i in range(start_layer, end_layer):
                # src self-attention
                self_attention_layer_outputs_en = roberta.encoder(
                    [middle_hidden_states_en, extended_attention_mask_en, head_mask, output_attentions,
                     output_hidden_states], training=training, start_layer=i, end_layer=i + 1, add_ffn=False
                )

                # src FFN layer
                ffn_layer_outputs_en = roberta.encoder(
                    [self_attention_layer_outputs_en[0], extended_attention_mask_en, head_mask, output_attentions,
                     output_hidden_states], training=training, start_layer=i, end_layer=i + 1, add_attention=False
                )
                middle_hidden_states_en = ffn_layer_outputs_en[0]

                # trg self-attention
                self_attention_layer_outputs = roberta.encoder(
                    [middle_hidden_states, extended_attention_mask, head_mask, output_attentions,
                     output_hidden_states], training=training, start_layer=i, end_layer=i + 1, add_ffn=False
                )
                # get attention scores
                cross_attention_layer_outputs = roberta.encoder(
                    [self_attention_layer_outputs[0], extended_attention_mask, self_attention_layer_outputs_en[0],
                     extended_attention_mask_en, head_mask, output_attentions, output_hidden_states],
                    training=training, start_layer=i, end_layer=i + 1, cross_attention=True, add_ffn=False
                )
                attention_score = cross_attention_layer_outputs[-1][0]  # (batch size, num_heads, seq_len_q, seq_len_k)
                attention_entropy = get_pair_entropy(attention_score, extended_attention_mask, attention_mask,
                                                     extended_attention_mask_en, attention_mask_en)
                attention_entropy = tf.reshape(attention_entropy, shape=(-1, 1))
                lamb = self.entropy_dense[i](attention_entropy)
                lamb = lambda_0 * self.lambda_act(lamb)
                lamb = tf.reshape(lamb, shape=(-1, 1, 1))

                # trg cross-attention
                cross_attention_layer_outputs = roberta.encoder(
                    [self_attention_layer_outputs[0], extended_attention_mask, self_attention_layer_outputs_en[0],
                     extended_attention_mask_en, head_mask, output_attentions, output_hidden_states],
                    training=training, start_layer=i, end_layer=i + 1, cross_attention=True, ratio=lamb
                )
                middle_hidden_states = cross_attention_layer_outputs[0]

            encoder_outputs = roberta.encoder(
                [middle_hidden_states, extended_attention_mask, head_mask, output_attentions, output_hidden_states],
                training=training, start_layer=end_layer
            )
            encoder_outputs_en = roberta.encoder(
                [middle_hidden_states_en, extended_attention_mask_en, head_mask, output_attentions, output_hidden_states],
                training=training, start_layer=end_layer
            )
            sequence_output = encoder_outputs[0]
            en_sequence_output = encoder_outputs_en[0]

            outputs = (encoder_outputs,) + (sequence_output,) + (en_sequence_output,)

        return outputs


def get_attention_entropy(attention_score, attention_mask_soft, attention_mask):
    """Get attention entropy based on attention score."""
    bz, n_heads, seq_len_q, seq_len_k = shape_list(attention_score)
    soft_attention_score = attention_score + attention_mask_soft  # (batch size, num_heads, seq_len_q, seq_len_k)
    soft_attention_score = tf.nn.softmax(soft_attention_score, axis=-1)
    masked_attention_score = tf.multiply(attention_score, attention_mask)
    soft_attention_score = tf.reshape(soft_attention_score, shape=[-1, seq_len_k])
    masked_attention_score = tf.reshape(masked_attention_score, shape=[-1, seq_len_k])
    attention_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=soft_attention_score, logits=masked_attention_score)
    attention_entropy = tf.reshape(attention_entropy, shape=[bz, n_heads, -1])  # (batch size, num_heads, seq_len_q)
    mean_attention_entropy = tf.reduce_mean(attention_entropy, axis=[1, 2])  # (batch size, )

    return mean_attention_entropy


def get_pair_entropy(attention_score, attention_mask_soft, attention_mask, attention_mask_soft_en, attention_mask_en):
    """Get pairwise attention entropy."""
    entropy1 = get_attention_entropy(attention_score, attention_mask_soft_en, attention_mask_en)
    attention_score = tf.transpose(attention_score, perm=[0, 1, 3, 2])
    entropy2 = get_attention_entropy(attention_score, attention_mask_soft, attention_mask)

    entropy = entropy1 + entropy2

    return entropy


def get_sequence_rep(hidden_states, attention_mask):
    """Get sequence representation from hidden states."""
    layer_hidden = hidden_states[:, 1:-1, :]
    bz, seq_len, hidden_dim = shape_list(layer_hidden)
    flat_hidden = tf.reshape(layer_hidden, shape=[-1, hidden_dim])
    flat_mask = tf.cast(tf.reshape(attention_mask[:, 2:], shape=[-1, 1]), dtype=tf.float32)
    masked_flat_hidden = flat_mask * flat_hidden
    masked_hidden = tf.reshape(masked_flat_hidden, shape=[bz, -1, hidden_dim])
    sent_rep = tf.reduce_mean(masked_hidden, axis=1)

    return sent_rep
