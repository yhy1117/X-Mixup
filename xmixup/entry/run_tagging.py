# coding=utf-8
""" Fine-tuning the library models for sequence classification."""


import logging
import os
import warnings
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from seqeval.metrics import f1_score, precision_score, recall_score

from xmixup.configuration.auto import AutoConfig
from xmixup.tokenization.auto import AutoTokenizer
from xmixup.tasks.auto_model import AutoModelForTask
from xmixup.trainers.trainer_tagging import TaggingTrainer as Trainer
from xmixup.trainers.training_args import TrainingArguments
from xmixup.utils.train_utils import EvalPrediction
from xmixup.utils.hf_argparser import HfArgumentParser
from xmixup.data.datasets.ner import TagProcessor
from xmixup.data.tasks.ner import convert_examples_to_features

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
        default="ner",
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
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
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

    # Prepare tagging task
    train_languages = data_args.train_languages.split(',')
    eval_languages = data_args.eval_languages.split(',')
    processor = TagProcessor(languages=eval_languages, train_languages=train_languages, task_name=data_args.task_name)
    label_list = processor.get_labels_from_file(data_args.data_dir, data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(label_list)}
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
            "token classification",
            model_args.model_name_or_path,
            from_pt=True,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Get datasets
    train_examples = processor.get_examples_with_trans(data_args.data_dir, add_back_trans=True)
    train_dataset = (
        convert_examples_to_features(train_examples, tokenizer, processor, data_args.max_seq_length,
                                     label_list=label_list, task=data_args.task_name, add_trans=True,
                                     add_back_trans=True, is_training=True)
        if training_args.do_train
        else None
    )

    eval_datasets = []
    for lang in eval_languages:
        eval_examples = processor.get_examples_with_trans(data_args.data_dir, set_type="dev", lang=lang)
        eval_dataset = (
            convert_examples_to_features(eval_examples, tokenizer, processor, data_args.max_seq_length,
                                         label_list=label_list, task=data_args.task_name, add_trans=True)
            if training_args.do_eval
            else None
        )
        eval_datasets.append(eval_dataset)

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[List], List[List]]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] == -1:
                    label_ids[i, j] = -100
                    warnings.warn(
                        "Using `-1` to mask the loss for the token is depreciated. Please use `-100` instead."
                    )
                if label_ids[i, j] != -100:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)

        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        eval_languages=eval_languages,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    # Predict
    if training_args.do_predict:
        test_results = []
        for lang in eval_languages:
            test_examples = processor.get_examples_with_trans(data_args.data_dir, set_type="test", lang=lang)
            test_dataset = convert_examples_to_features(test_examples, tokenizer, processor, data_args.max_seq_length,
                                             label_list=label_list, task=data_args.task_name, add_trans=True)

            predictions, label_ids, metrics = trainer.predict(test_dataset)
            preds_list, labels_list = align_predictions(predictions, label_ids)
            # report = classification_report(labels_list, preds_list)
            if "eval_f1" in metrics:
                test_results.append(metrics["eval_f1"])
            # Save test results
            logger.info("\nLanguage {} result: {}".format(lang, metrics))
            output_test_results_file = os.path.join(training_args.output_dir, lang + "_test_results.txt")
            with tf.io.gfile.GFile(output_test_results_file, "w") as writer:
                writer.write("Language %s: %s\n" % (lang, metrics))

            # Save predictions
            output_test_predictions_file = os.path.join(training_args.output_dir, lang + "_test_predictions.txt")
            processor.save_predictions(output_test_predictions_file, data_args.data_dir, lang, preds_list)

        mean_results = tf.math.reduce_mean(test_results)
        logger.info("Test mean results: {}".format(mean_results))
        output_test_results_file = os.path.join(training_args.output_dir, "ave_test_results.txt")
        with tf.io.gfile.GFile(output_test_results_file, "w") as writer:
            writer.write("Test mean results: {}".format(mean_results))


if __name__ == "__main__":
    main()
