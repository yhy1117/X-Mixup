import tensorflow as tf
from xmixup.models.roberta import RobertaMainLayer
from xmixup.models.pretrain_model import RobertaPreTrainedModel
from xmixup.tasks.loss import SequenceClassificationLoss
from xmixup.tokenization.utils import BatchEncoding
from xmixup.configuration.xlmr import XLMRobertaConfig
from xmixup.tasks.layer_mixup import XMixup
from xmixup.tasks.mbert.sequence_classification import RobertaClassificationHead


class XMixupForSequenceClassification(RobertaPreTrainedModel, SequenceClassificationLoss):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    config_class = XLMRobertaConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.roberta = RobertaMainLayer(config, name="roberta")
        self.mixup = XMixup(config, name="x_mixup")
        self.classifier = RobertaClassificationHead(config, name="classifier")
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
        alpha=0.4
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
            logits = self.classifier(sequence_output, training=training)
            outputs = (logits,) + (logits,) + outputs[2:]

            if labels is not None:
                loss = self.compute_loss(labels, logits)
                outputs = (loss,) + outputs
        else:
            sequence_output = outputs[1]
            en_sequence_output = outputs[2]

            # representation MSE loss
            src_rep = en_sequence_output[:, 0, :] # [batch_size, hidden_dims]
            trg_rep = sequence_output[:, 0, :]

            # logits KL loss
            src_logits = self.classifier(en_sequence_output, training=training)
            trg_logits = self.classifier(sequence_output, training=training)

            # task loss
            src_task_loss = self.compute_loss(labels, src_logits)
            trg_task_loss = self.compute_loss(labels, trg_logits)
            src_soft_logits = tf.nn.softmax(src_logits, axis=-1)
            trg_soft_logits = tf.nn.softmax(trg_logits, axis=-1)

            task_loss = alpha * src_task_loss + (1 - alpha) * trg_task_loss

            # consistent loss
            consist_kl_loss = self.kl_loss(src_soft_logits, trg_soft_logits)
            consist_mse_loss = self.mse_loss(src_rep, trg_rep)
            consist_loss = consist_kl_loss + consist_mse_loss

            loss = task_loss + consist_loss

            outputs = (loss,) + (trg_logits,) + (src_logits,) + (task_loss,) + (consist_loss,) + (src_task_loss,) + \
                      (trg_task_loss,) + (consist_kl_loss,) + (consist_mse_loss,)

        return outputs  # (loss), logits, (hidden_states), (attentions)
