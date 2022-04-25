"""Loss function for different tasks."""
import tensorflow as tf
from xmixup.utils.model_utils import shape_list


class SequenceClassificationLoss:
    @staticmethod
    def compute_loss(labels, logits):
        if shape_list(logits)[1] == 1:
            loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        else:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )

        return loss_fn(labels, logits)


class TokenClassificationLoss:
    @staticmethod
    def compute_loss(labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100
        # are taken into account as loss
        active_loss = tf.reshape(labels, (-1,)) != -100
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

        return loss_fn(labels, reduced_logits)


class QuestionAnsweringLoss:
    @staticmethod
    def compute_loss(labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        start_loss = loss_fn(labels["start_position"], logits[0])
        end_loss = loss_fn(labels["end_position"], logits[1])

        return (start_loss + end_loss) / 2.0
