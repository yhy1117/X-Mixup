"""Tensorflow trainers class for qa tasks."""

import logging
import math
import os
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from xmixup.models.pretrain_model import PreTrainedModel
from xmixup.models.optimization import GradientAccumulator
from xmixup.utils.train_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, is_wandb_available, set_seed
from xmixup.trainers.training_args import TrainingArguments
from xmixup.trainers.trainer import Trainer
from xmixup.data.tasks.qa import SquadResult
from xmixup.metrics.squad import compute_predictions_logits, squad_evaluate


if is_wandb_available():
    pass


logger = logging.getLogger(__name__)


class QATrainer(Trainer):
    """
    Trainer for QA tasks.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_args: Optional[dict]
    train_dataset: Optional[tf.data.Dataset]
    eval_dataset: Optional[tf.data.Dataset]
    eval_examples: Optional[list]
    eval_features: Optional[list]
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
        data_args: Optional[dict] = None,
        train_dataset: Optional[tf.data.Dataset] = None,
        eval_dataset=None,
        eval_languages=None,
        eval_examples=None,
        eval_features=None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional[tf.summary.SummaryWriter] = None,
        optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None,
    ):
        super().__init__(model, args, train_dataset, eval_dataset, compute_metrics, prediction_loss_only, tb_writer, optimizers)
        self.data_args = data_args
        self.eval_languages = eval_languages
        self.eval_examples = eval_examples
        self.eval_features = eval_features

    def get_eval_tfdataset(self, eval_dataset: Optional[tf.data.Dataset] = None):
        """
        Returns the evaluation :class:`~tf.data.Dataset`.

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                If provided, will override `self.eval_dataset`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        eval_tfdatasets = []
        for dataset in eval_dataset:
            ds = (
                dataset.cache()
                .batch(self.args.eval_batch_size, drop_remainder=self.args.dataloader_drop_last)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )
            eval_tfdatasets.append(self.args.strategy.experimental_distribute_dataset(ds))

        return eval_tfdatasets

    def _prediction_loop(
        self, dataset, description: str, prediction_loss_only: Optional[bool] = None,
            eval_examples=None, eval_features=None, test_features=None
    ):
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only
        if description == "Prediction":
            logger.info("***** Running %s *****", description)
            logger.info("  Batch size = %d", self.args.eval_batch_size)

        all_result = []
        losses: np.ndarray = None
        start_logits_list: np.ndarray = None
        end_logits_list: np.ndarray = None

        step: int = 1

        # Reset the past mems state at the beginning of the evaluation if necessary.
        if self.args.past_index >= 0:
            self._past = None

        if description == "Prediction":
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
            loss, logits = self._evaluate_steps(features, labels)
            loss = tf.reduce_mean(loss)
            if losses is None:
                losses = loss.numpy()
            else:
                losses = np.append(losses, loss.numpy())

            if not prediction_loss_only:
                if isinstance(logits, list):
                    start_logits = logits[0]
                    end_logits = logits[1]
                else:
                    start_logits = None
                    end_logits = None

                if self.args.n_replicas > 1:
                    for start in start_logits.values:
                        if start_logits_list is None:
                            start_logits_list = start.numpy()
                        else:
                            start_logits_list = np.append(start_logits_list, start.numpy(), axis=0)

                    for end in end_logits.values:
                        if end_logits_list is None:
                            end_logits_list = end.numpy()
                        else:
                            end_logits_list = np.append(end_logits_list, end.numpy(), axis=0)

                else:
                    if start_logits_list is None:
                        start_logits_list = start_logits.numpy()
                    else:
                        start_logits_list = np.append(start_logits_list, start_logits.numpy(), axis=0)
                    if end_logits_list is None:
                        end_logits_list = end_logits.numpy()
                    else:
                        end_logits_list = np.append(end_logits_list, end_logits.numpy(), axis=0)

            step += 1

        if description == "Prediction":
            assert len(test_features) == len(start_logits_list)
            assert len(start_logits_list) == len(end_logits_list)
            for idx, (start, end) in enumerate(zip(start_logits_list, end_logits_list)):
                result = SquadResult(unique_id=test_features[idx].unique_id, start_logits=start, end_logits=end)
                all_result.append(result)
            return all_result

        elif description == "Evaluation":
            assert len(eval_features) == len(start_logits_list)
            assert len(start_logits_list) == len(end_logits_list)
            for idx, (start, end) in enumerate(zip(start_logits_list, end_logits_list)):
                result = SquadResult(unique_id=eval_features[idx].unique_id, start_logits=start, end_logits=end)
                all_result.append(result)
            n_best_size = self.data_args["n_best_size"]
            max_answer_length = self.data_args["max_answer_length"]
            tokenizer = self.data_args["tokenizer"]
            do_lower_case = self.data_args["do_lower_case"]

            verbose_logging = self.data_args["verbose_logging"]
            version_2_with_negative = self.data_args["version_2_with_negative"]
            null_score_diff_threshold = self.data_args["null_score_diff_threshold"]
            predictions = compute_predictions_logits(eval_examples, eval_features, all_result, n_best_size,
                                                 max_answer_length, do_lower_case, verbose_logging, version_2_with_negative,
                                                     null_score_diff_threshold, tokenizer, is_predict=False)
            results = squad_evaluate(eval_examples, predictions)

            extract_match = results["exact_match"]
            f1 = results["f1"]
            total = results["total"]
            metrics = {"eval_f1": f1, "eval_em": extract_match, "eval_total": total}

            losses = np.mean(losses)
            metrics["eval_loss"] = losses

            if self.args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            return metrics

    def sample_p(self):
        k = self.args.p_k
        p = k / (k + np.exp(self.global_step / k))
        sample_p = random.random()

        return sample_p <= p

    def evaluate(self, eval_dataset: Optional[tf.data.Dataset] = None, final_eval: Optional[bool] = False) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        Args:
            eval_dataset (:class:`~tf.data.Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        logs = {}

        eval_ds = self.get_eval_tfdataset(eval_dataset)

        output_metrics = {}
        for lang, ds, examples, features in zip(self.eval_languages, eval_ds, self.eval_examples, self.eval_features):
            output = self._prediction_loop(ds, description="Evaluation", eval_examples=examples, eval_features=features)
            for key in list(output.keys()):
                if key not in output_metrics:
                    output_metrics[key] = output[key]
                else:
                    output_metrics[key] += output[key]

            logs[f"{lang}_f1"] = output["eval_f1"]
            logs[f"{lang}_em"] = output["eval_em"]

        n_langs = len(self.eval_languages)
        for key in list(output_metrics.keys()):
            output_metrics[key] = output_metrics[key] / n_langs

        logs["ave_f1"] = output_metrics["eval_f1"]
        logs["ave_em"] = output_metrics["eval_em"]
        logs["epoch"] = self.epoch_logging

        self._log(logs)

        return output_metrics

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
            ckpt = tf.train.Checkpoint(model=self.model)
            self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=self.args.save_total_limit)

            if self.model.ckpt_manager.latest_checkpoint:
                epochs_trained = self.global_step // (self.num_train_examples // self.args.gradient_accumulation_steps) + 1
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
                    if "eval_f1" in eval_metrics:
                        current_result = eval_metrics["eval_f1"]
                        if "eval_em" in eval_metrics:
                            current_result += eval_metrics["eval_em"]
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
                    # and self.args.logging_first_step
                ):
                    logs = {}
                    logs["loss"] = training_loss[0].numpy()
                    logs["task_loss"] = training_loss[1].numpy()
                    logs["consist_loss"] = training_loss[2].numpy()
                    logs["src_task_loss"] = training_loss[3].numpy()
                    logs["trg_task_loss"] = training_loss[4].numpy()
                    logs["consist_rep_loss"] = training_loss[5].numpy()
                    logs["consist_ans_loss"] = training_loss[6].numpy()
                    logs["learning_rate"] = lr_scheduler(self.global_step).numpy()
                    logs["epoch"] = self.epoch_logging

                    self._log(logs)

                if self.args.max_steps > 0 and self.global_step % self.args.max_steps == 0:
                    break

        if not self.args.evaluate_during_training:
            # Save the final model
            ckpt_save_path = self.model.ckpt_manager.save()
            logger.info("Saving checkpoint for step {} at {}".format(self.global_step, ckpt_save_path))

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

    def _accumulate_gradients(self, per_replica_features, per_replica_labels):
        """Accumulates the gradients across all the replica."""
        per_replica_loss, per_replica_detailed_loss = self.args.strategy.experimental_run_v2(
            self._forward, args=(per_replica_features, per_replica_labels)
        )

        try:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=0)
            reduced_task_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[0], axis=0)
            reduced_consist_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[1], axis=0)
            reduced_src_task_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[2], axis=0)
            reduced_trg_task_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[3], axis=0)
            reduced_consist_rep_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_detailed_loss[4], axis=0)
            reduced_ans_rep_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                                                 per_replica_detailed_loss[5], axis=0)

            return reduced_loss, reduced_task_loss, reduced_consist_loss, reduced_src_task_loss, reduced_trg_task_loss, \
                   reduced_consist_rep_loss, reduced_ans_rep_loss

        except ValueError:
            reduced_loss = self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

            return reduced_loss

    def _forward(self, features, labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss, _, per_example_detailed_loss = self._run_model(features, labels, True)
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
            loss, logits, task_loss, consist_loss, src_task_loss, trg_task_loss, consist_rep_loss, consist_ans_loss = outputs
        else:
            loss, logits = outputs[:2]
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        loss += sum(self.model.losses) * (1.0 / self.args.n_replicas)
        if training:
            return loss, logits, [task_loss, consist_loss, src_task_loss, trg_task_loss, consist_rep_loss,
                                  consist_ans_loss]
        else:
            return loss, logits

    def predict(self, test_dataset: tf.data.Dataset, test_features=None):
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:class:`~tf.data.Dataset`):
                Dataset to run the predictions on.
        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """
        test_ds = self.get_test_tfdataset(test_dataset)

        return self._prediction_loop(test_ds, description="Prediction", test_features=test_features)
