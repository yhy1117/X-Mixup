"""Tensorflow trainers class for classification tasks."""

import logging
import math
import os
from typing import Callable, Dict, Optional, Tuple
import random
import numpy as np
import tensorflow as tf

from xmixup.models.pretrain_model import PreTrainedModel
from xmixup.trainers.training_args import TrainingArguments
from xmixup.utils.train_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput
from xmixup.trainers.trainer import Trainer


logger = logging.getLogger(__name__)


class ClassificationTrainer(Trainer):
    """
    Trainer for classification tasks.
    """

    model: PreTrainedModel
    args: TrainingArguments
    train_dataset: Optional[tf.data.Dataset]
    eval_dataset: Optional[tf.data.Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional[tf.summary.SummaryWriter] = None
    optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None
    global_step: Optional[int] = None
    epoch_logging: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Optional[tf.data.Dataset] = None,
        eval_dataset: Optional[tf.data.Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional[tf.summary.SummaryWriter] = None,
        optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None,
    ):
        super().__init__(model, args, train_dataset, eval_dataset, compute_metrics, prediction_loss_only, tb_writer, optimizers)

    @tf.function
    def _evaluate_steps(self, per_replica_features, per_replica_labels):
        """
        One step evaluation across replica.
        Args:
          per_replica_features: the batched features.
          per_replica_labels: the batched labels.
        Returns:
          The loss corresponding to the given batch.
        """
        per_replica_loss, per_replica_trg_logits, per_replica_src_logits = self.args.strategy.experimental_run_v2(
            self._run_model, args=(per_replica_features, per_replica_labels, False)
        )

        try:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0)
        except ValueError:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

        return reduced_loss, per_replica_trg_logits, per_replica_src_logits

    def _prediction_loop(
        self, dataset: tf.data.Dataset, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        logger.info("***** Running %s *****", description)
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        label_ids: np.ndarray = None
        trg_preds: np.ndarray = None
        src_preds: np.ndarray = None

        step: int = 1

        # Reset the past mems state at the beginning of the evaluation if necessary.
        if self.args.past_index >= 0:
            self._past = None

        if description == "Prediction" or description == "Final evaluation":
            # Restore from latest checkpoint
            folder = os.path.join(self.args.output_dir, PREFIX_CHECKPOINT_DIR)
            ckpt = tf.train.Checkpoint(model=self.model)
            self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=self.args.save_total_limit)

            if self.model.ckpt_manager.latest_checkpoint:
                logger.info(
                    "Checkpoint file %s found and restoring from checkpoint", self.model.ckpt_manager.latest_checkpoint
                )
                ckpt.restore(self.model.ckpt_manager.latest_checkpoint).expect_partial()

        for features, labels in dataset:
            step = tf.convert_to_tensor(step, dtype=tf.int64)
            loss, trg_logits, src_logits = self._evaluate_steps(features, labels)
            loss = tf.reduce_mean(loss)

            label_ids, trg_preds = self.merge_results(trg_logits, labels, label_ids, trg_preds, prediction_loss_only, merge_label=True)
            _, src_preds = self.merge_results(src_logits, labels, label_ids, src_preds, prediction_loss_only)

            step += 1

        preds = (src_preds + trg_preds) / 2

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        metrics["eval_loss"] = loss.numpy()

        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def merge_results(self, logits, labels, label_ids, preds, prediction_loss_only, merge_label=False):
        if not prediction_loss_only:
            if isinstance(logits, tuple):
                logits = logits[0]

            if isinstance(labels, tuple):
                labels = labels[0]

            if self.args.n_replicas > 1:
                for val in logits.values:
                    if preds is None:
                        preds = val.numpy()
                    else:
                        preds = np.append(preds, val.numpy(), axis=0)

                if merge_label:
                    for val in labels.values:
                        if label_ids is None:
                            label_ids = val.numpy()
                        else:
                            label_ids = np.append(label_ids, val.numpy(), axis=0)
            else:
                if preds is None:
                    preds = logits.numpy()
                else:
                    preds = np.append(preds, logits.numpy(), axis=0)

                if merge_label:
                    if label_ids is None:
                        label_ids = labels.numpy()
                    else:
                        label_ids = np.append(label_ids, labels.numpy(), axis=0)

        return label_ids, preds

    def sample_p(self):
        k = self.args.p_k
        p = k / (k + np.exp(self.global_step / k))
        sample_p = random.random()

        return sample_p <= p

    def train(self) -> None:
        """
        Train method to train the model.
        """
        train_ds = self.get_train_tfdataset()

        if self.args.debug:
            tf.summary.trace_on(graph=True, profiler=True)

        self.gradient_accumulator.reset()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            steps_per_epoch = self.args.max_steps
        else:
            if self.args.dataloader_drop_last:
                approx = math.floor
            else:
                approx = math.ceil

            steps_per_epoch = approx(
                self.num_train_examples / (self.args.train_batch_size * self.args.gradient_accumulation_steps)
            )
            t_total = steps_per_epoch * self.args.num_train_epochs

        with self.args.strategy.scope():
            optimizer, lr_scheduler = self.get_optimizers(num_training_steps=t_total)
            iterations = optimizer.iterations
            self.global_step = iterations.numpy()
            folder = os.path.join(self.args.output_dir, PREFIX_CHECKPOINT_DIR)
            ckpt = tf.train.Checkpoint(optimizer=optimizer, model=self.model)
            self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=self.args.save_total_limit)

            if self.model.ckpt_manager.latest_checkpoint:
                epochs_trained = self.global_step // (self.num_train_examples // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    self.num_train_examples // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
                logger.info(
                    "Checkpoint file %s found and restoring from checkpoint", self.model.ckpt_manager.latest_checkpoint
                )

                ckpt.restore(self.model.ckpt_manager.latest_checkpoint).expect_partial()
            else:
                epochs_trained = 1

        tf.summary.experimental.set_step(iterations)

        epochs = 1 if self.args.max_steps > 0 else self.args.num_train_epochs

        if self.args.fp16:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
            tf.keras.mixed_precision.experimental.set_policy(policy)

        with self.tb_writer.as_default():
            tf.summary.text("args", self.args.to_json_string())

        self.tb_writer.flush()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_train_examples)
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d", self.args.train_batch_size
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.best_eval_step = 0
        self.best_eval_result = 0.0
        for epoch_iter in range(epochs_trained, int(epochs + 1)):
            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None
            for step, training_loss in enumerate(self._training_steps(train_ds, optimizer)):
                self.global_step = iterations.numpy()
                self.epoch_logging = epoch_iter - 1 + (step + 1) / steps_per_epoch

                if self.args.debug:
                    logs = {}
                    logs["loss"] = training_loss[0].numpy()
                    logs["epoch"] = self.epoch_logging

                    self._log(logs)

                if self.global_step == 1 and self.args.debug:
                    with self.tb_writer.as_default():
                        tf.summary.trace_export(
                            name="training", step=self.global_step, profiler_outdir=self.args.logging_dir
                        )

                if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if "eval_acc" in eval_metrics:
                        current_result = eval_metrics["eval_acc"]
                    elif "eval_f1" in eval_metrics:
                        current_result = eval_metrics["eval_f1"]
                    else:
                        current_result = 0
                    if current_result > self.best_eval_result:
                        self.best_eval_result = current_result
                        self.best_eval_step = step
                        if self.global_step % self.args.save_steps == 0:
                            ckpt_save_path = self.model.ckpt_manager.save()
                            logger.info("Saving checkpoint for step {} at {}".format(self.global_step, ckpt_save_path))

                if (
                    self.global_step % self.args.logging_steps == 0
                    or self.global_step == 1
                    and self.args.logging_first_step
                ):
                    logs = {}
                    logs["loss"] = training_loss[0].numpy()
                    logs["task_loss"] = training_loss[1].numpy()
                    logs["consist_loss"] = training_loss[2].numpy()
                    logs["src_task_loss"] = training_loss[3].numpy()
                    logs["trg_task_loss"] = training_loss[4].numpy()
                    logs["consist_kl_loss"] = training_loss[5].numpy()
                    logs["consist_mse_loss"] = training_loss[6].numpy()
                    logs["learning_rate"] = lr_scheduler(self.global_step).numpy()
                    logs["epoch"] = self.epoch_logging

                    self._log(logs)

                if self.args.max_steps > 0 and self.global_step % self.args.max_steps == 0:
                    break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

    def _training_steps(self, ds, optimizer):
        """
        Returns a generator over training steps (i.e. parameters update).
        """
        for i, loss in enumerate(self._accumulate_next_gradients(ds)):
            if i % self.args.gradient_accumulation_steps == 0:
                self._apply_gradients(optimizer)
                yield loss

    def _accumulate_gradients(self, per_replica_features, per_replica_labels):
        """Accumulates the gradients across all the replica."""
        per_replica_loss, per_replica_detailed_loss = self.args.strategy.experimental_run_v2(
            self._forward, args=(per_replica_features, per_replica_labels)
        )

        try:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0)
            reduced_task_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[0], axis=0)
            reduced_consist_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[1], axis=0)
            reduced_src_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[2],
                                                             axis=0)
            reduced_trg_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[3],
                                                             axis=0)
            reduced_kl_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[4],
                                                             axis=0)
            reduced_mse_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[5],
                                                        axis=0)

            return reduced_loss, reduced_task_loss, reduced_consist_loss, reduced_src_loss, reduced_trg_loss, \
                   reduced_kl_loss, reduced_mse_loss

        except ValueError:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

            return reduced_loss

    def _forward(self, features, labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss, _, _, per_example_detailed_loss = self._run_model(features, labels, True)
        gradients = tf.gradients(per_example_loss, self.model.trainable_variables)
        gradients = [
            g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)
        ]

        self.gradient_accumulator(gradients)

        return per_example_loss, per_example_detailed_loss

    def _run_model(self, features, labels, training):
        """
        Computes the loss of the given features and labels pair.
        Args:
          features: the batched features.
          labels: the batched labels.
          training: run the model in training mode or not
        """
        if self.args.past_index >= 0 and getattr(self, "_past", None) is not None:
            features["mems"] = self._past

        if training:
            sample_gold = self.sample_p()
        else:
            sample_gold = True

        if isinstance(labels, (dict)):
            outputs = self.model(features, training=training, **labels, mixup_inference=self.args.mixup_inference,
                                 start_layer=self.args.start_layer, end_layer=self.args.end_layer, alpha=self.args.alpha,
                                 sample_gold=sample_gold)
        else:
            outputs = self.model(features, labels=labels, training=training, mixup_inference=self.args.mixup_inference,
                                 start_layer=self.args.start_layer, end_layer=self.args.end_layer, alpha=self.args.alpha,
                                 sample_gold=sample_gold)
        if training:
            loss, trg_logits, src_logits, task_loss, consist_loss, src_task_loss, trg_task_loss, consist_kl_loss, \
            consist_mse_loss = outputs
        else:
            loss, trg_logits, src_logits = outputs[:3]
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        loss += sum(self.model.losses) * (1.0 / self.args.n_replicas)
        if training:
            return loss, trg_logits, src_logits, \
                   [task_loss, consist_loss, src_task_loss, trg_task_loss, consist_kl_loss, consist_mse_loss]
        else:
            return loss, trg_logits, src_logits
