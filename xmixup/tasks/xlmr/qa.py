import tensorflow as tf
from xmixup.models.roberta import RobertaMainLayer
from xmixup.models.pretrain_model import RobertaPreTrainedModel
from xmixup.tasks.loss import QuestionAnsweringLoss
from xmixup.utils.model_utils import get_initializer
from xmixup.tokenization.utils import BatchEncoding
from xmixup.configuration.xlmr import XLMRobertaConfig
from xmixup.tasks.layer_mixup import XMixup, get_sequence_rep
from xmixup.tasks.mbert.qa import get_outputs


class XMixupForQuestionAnswering(RobertaPreTrainedModel, QuestionAnsweringLoss):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"lm_head"]
    _keys_to_ignore_on_load_missing = [r"dropout"]

    config_class = XLMRobertaConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = 2

        self.roberta = RobertaMainLayer(config, name="roberta")
        self.mixup = XMixup(config, name="x_mixup")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
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
        start_positions=None,
        end_positions=None,
        start_positions_en=None,
        end_positions_en=None,
        training=False,
        mixup_inference=False,
        start_layer=1,
        end_layer=20,
        sample_gold=True,
        alpha=0.2
    ):
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Return:
        :obj:`tuple(tf.Tensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        start_scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
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
            start_positions = inputs[8] if len(inputs) > 8 else start_positions
            end_positions = inputs[9] if len(inputs) > 9 else end_positions
            if len(inputs) > 8:
                inputs = inputs[:8]
        elif isinstance(labels, (dict, BatchEncoding)):
            start_positions = labels.pop("start_positions", start_positions)
            end_positions = labels.pop("end_positions", end_positions)

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
            logits = self.qa_outputs(sequence_output)
            outputs = get_outputs(logits)

            if start_positions is not None and end_positions is not None:
                labels = {"start_position": start_positions}
                labels["end_position"] = end_positions
                loss = self.compute_loss(labels, outputs[0])
                outputs = (loss,) + outputs
        else:
            sequence_output = outputs[1]
            en_sequence_output = outputs[2]

            # Task loss
            trg_logits = self.qa_outputs(sequence_output)
            src_logits = self.qa_outputs(en_sequence_output)
            trg_logits_outputs = get_outputs(trg_logits)
            src_logits_outputs = get_outputs(src_logits)
            trg_labels = {"start_position": start_positions, "end_position": end_positions}
            src_labels = {"start_position": start_positions_en, "end_position": end_positions_en}
            src_task_loss = self.compute_loss(src_labels, src_logits_outputs[0])
            trg_task_loss = self.compute_loss(trg_labels, trg_logits_outputs[0])
            if training and sample_gold:
                task_loss = alpha * src_task_loss + (1 - alpha) * trg_task_loss
            else:
                task_loss = trg_task_loss
            # Consist loss
            attention_mask = inputs.get("attention_mask", attention_mask)
            attention_mask_en = inputs.get("attention_mask_en", attention_mask)
            src_rep = get_sequence_rep(en_sequence_output, attention_mask_en)
            trg_rep = get_sequence_rep(sequence_output, attention_mask)
            consist_rep_loss = self.mse_loss(src_rep, trg_rep)
            consist_loss = consist_rep_loss

            loss = task_loss + consist_loss

            outputs = (loss,) + (trg_logits_outputs[0],) + (task_loss,) + (consist_loss,) + (src_task_loss,) + \
                      (trg_task_loss,) + (consist_rep_loss,) + (consist_rep_loss,)

        return outputs  # (loss), ([start_logits, end_logits])
