import tensorflow as tf
from xmixup.models.roberta import RobertaMainLayer
from xmixup.models.pretrain_model import RobertaPreTrainedModel
from xmixup.tasks.loss import TokenClassificationLoss
from xmixup.utils.model_utils import get_initializer
from xmixup.tokenization.utils import BatchEncoding
from xmixup.configuration.xlmr import XLMRobertaConfig
from xmixup.tasks.layer_mixup import XMixup, get_sequence_rep
from xmixup.tasks.mbert.token_classification import sequence_subword_pooling


class XMixupForTokenClassification(RobertaPreTrainedModel, TokenClassificationLoss):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    config_class = XLMRobertaConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = RobertaMainLayer(config, name="roberta")
        self.mixup = XMixup(config, name="x_mixup")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        self.mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE, name="mse_loss")
        self.kl_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE, name='kl_divergence')

    def call(
        self,
        inputs=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        training=False,
        mixup_inference=False,
        start_layer=1,
        end_layer=20,
        sample_gold=True,
        alpha=0.8,
        use_subword_pooling=True
    ):
        r"""
    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        logits (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
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
        if isinstance(inputs, (tuple, list)):
            labels = inputs[8] if len(inputs) > 8 else labels
            if len(inputs) > 8:
                inputs = inputs[:8]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop("labels", labels)

        outputs = self.mixup.manifold_mixup(
                self.roberta,
                inputs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                training=training,
                mixup_inference=mixup_inference,
                start_layer=start_layer,
                end_layer=end_layer,
                sample_gold=sample_gold
            )

        if not training and not mixup_inference:
            sequence_output = outputs[0]
            if use_subword_pooling:
                pooling_ids = inputs.get("pooling_ids", None)
                if pooling_ids is not None:
                    sequence_output = sequence_subword_pooling(sequence_output, pooling_ids)
            sequence_output = self.dropout(sequence_output, training=training)
            logits = self.classifier(sequence_output)
            outputs = (logits,) + outputs[2:]

            if labels is not None:
                loss = self.compute_loss(labels, logits)
                outputs = (loss,) + outputs
        else:
            sequence_output = outputs[1]
            en_sequence_output = outputs[2]
            if use_subword_pooling:
                if training:
                    pooling_ids = inputs.pop("soft_pooling_ids", None)
                else:
                    pooling_ids = inputs.pop("pooling_ids", None)
                if pooling_ids is not None:
                    sequence_output = sequence_subword_pooling(sequence_output, pooling_ids)
            sequence_output = self.dropout(sequence_output, training=training)
            trg_logits = self.classifier(sequence_output)

            # task loss
            if training:
                soft_label_ids = inputs.pop("soft_label_ids", None)  # [bz, seq_len, num_labels]
                soft_label_sum = tf.reduce_sum(soft_label_ids, axis=-1)
                active_soft_loss = tf.reshape(soft_label_sum, (-1,)) > 0.5
                reduced_soft_logits = tf.boolean_mask(tf.reshape(trg_logits, (-1, self.num_labels)), active_soft_loss)
                reduced_soft_label = tf.boolean_mask(tf.reshape(soft_label_ids, (-1, self.num_labels)), active_soft_loss)
                trg_soft_logits = tf.nn.softmax(reduced_soft_logits, axis=-1)
                trg_task_loss = tf.reduce_mean(-tf.reduce_sum(reduced_soft_label * tf.math.log(trg_soft_logits + 1e-6),
                                                              axis=-1), keepdims=True)
                if sample_gold:
                    if use_subword_pooling:
                        pooling_ids = inputs.pop("pooling_ids", None)
                        if pooling_ids is not None:
                            en_sequence_output = sequence_subword_pooling(en_sequence_output, pooling_ids)
                    en_sequence_output = self.dropout(en_sequence_output, training=training)
                    src_logits = self.classifier(en_sequence_output)
                    src_task_loss = tf.reduce_mean(self.compute_loss(labels, src_logits), keepdims=True)
                    task_loss = alpha * src_task_loss + (1 - alpha) * trg_task_loss
                else:
                    task_loss = trg_task_loss
            else:
                task_loss = tf.reduce_mean(self.compute_loss(labels, trg_logits), keepdims=True)
                trg_task_loss = task_loss

            # consistent loss
            attention_mask = inputs.get("attention_mask", attention_mask)
            attention_mask_en = inputs.get("attention_mask_en", attention_mask)
            src_rep = get_sequence_rep(en_sequence_output, attention_mask_en)
            trg_rep = get_sequence_rep(sequence_output, attention_mask)
            consist_loss = tf.reduce_mean(self.mse_loss(src_rep, trg_rep), keepdims=True)

            loss = task_loss + consist_loss

            outputs = (loss,) + (trg_logits,) + (task_loss,) + (consist_loss,) + (trg_task_loss,)

        return outputs  # (loss), logits, (hidden_states), (attentions)
