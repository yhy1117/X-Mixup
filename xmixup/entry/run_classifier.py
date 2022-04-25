""" Fine-tuning the library models for sequence classification."""


import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from xmixup.configuration.auto import AutoConfig
from xmixup.tokenization.auto import AutoTokenizer
from xmixup.tasks.auto_model import AutoModelForTask
from xmixup.trainers.trainer_classification import ClassificationTrainer as Trainer
from xmixup.trainers.training_args import TrainingArguments
from xmixup.utils.train_utils import EvalPrediction
from xmixup.utils.hf_argparser import HfArgumentParser
from xmixup.data.tasks.classification import convert_examples_to_features


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    train_languages: str = field(
        metadata={"help": "The languages used for training, split by ','."}
    )
    eval_languages: str = field(
        metadata={"help": "The language used for evaluation, split by ','."}
    )
    task_name: str = field(
        default="xnli",
        metadata={"help": "The task name."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        tf.io.gfile.exists(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(
        "n_replicas: %s, distributed training: %s, 16-bits training: %s",
        training_args.n_replicas,
        bool(training_args.n_replicas > 1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Prepare text Classification task
    train_languages = data_args.train_languages.split(',')
    eval_languages = data_args.eval_languages.split(',')

    if "xnli" in data_args.task_name:
        from xmixup.data.datasets.xnli import Processor
    else:
        from xmixup.data.datasets.pawsx import Processor
    from xmixup.metrics.accuracy import compute_metrics
    processor = Processor(task_name=data_args.task_name, languages=eval_languages, train_languages=train_languages)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    config.n_langs = len(processor.get_languages())

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    with training_args.strategy.scope():
        model = AutoModelForTask.from_pretrained(
            "sequence classification",
            model_args.model_name_or_path,
            from_pt=True,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Get datasets
    train_examples = processor.get_train_examples(data_args.data_dir, add_trans=True, add_back_trans=True)
    train_dataset = (
        convert_examples_to_features(train_examples, tokenizer, processor, data_args.max_seq_length,
                                     label_list=label_list, is_training=True, add_back_trans=True)
        if training_args.do_train
        else None
    )

    eval_examples = processor.get_dev_examples(data_args.data_dir, add_trans=True)
    eval_dataset = (
        convert_examples_to_features(eval_examples, tokenizer, processor, data_args.max_seq_length,
                                     label_list=label_list, mixup_inference=True)
        if training_args.do_eval
        else None
    )

    # Initialize our Trainer
    def compute_metrics_for_dataset(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)

        return compute_metrics(preds, p.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_for_dataset,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate(final_eval=True)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")

        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")

            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    # Predict
    if training_args.do_predict:
        test_results = []
        for lang in eval_languages:
            test_examples = processor.get_test_examples_by_lang(data_args.data_dir, lang, add_trans=True)
            test_dataset = (
                convert_examples_to_features(test_examples, tokenizer, processor, data_args.max_seq_length,
                                             label_list=label_list, mixup_inference=True)
                if training_args.do_predict
                else None
            )
            predictions, label_ids, metrics = trainer.predict(test_dataset)
            pred_ids = np.argmax(predictions, axis=1)
            report = compute_metrics(pred_ids, label_ids)
            logger.info("\nLanguage {} result: {}".format(lang, report))
            if "acc" in report:
                test_results.append(report["acc"])
            # Save test results
            output_test_results_file = os.path.join(training_args.output_dir, lang + "_test_results.txt")
            with tf.io.gfile.GFile(output_test_results_file, "w") as writer:
                writer.write("Language %s accuracy: %s\n" % (lang, report["acc"]))

            # Save predictions
            output_test_predictions_file = os.path.join(training_args.output_dir, lang + "_test_predictions.txt")
            processor.save_predictions(output_test_predictions_file, data_args.data_dir, lang, pred_ids)

        mean_results = tf.math.reduce_mean(test_results)
        logger.info("Test mean results: {}".format(mean_results))
        output_test_results_file = os.path.join(training_args.output_dir, "ave_test_results.txt")
        with tf.io.gfile.GFile(output_test_results_file, "w") as writer:
            writer.write("Test mean results: {}".format(mean_results))


if __name__ == "__main__":
    main()
