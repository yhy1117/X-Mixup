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
""" Fine-tuning the library models for question-answering."""


import logging
import os
from dataclasses import dataclass, field
from typing import Optional
import tensorflow as tf

from xmixup.configuration.auto import AutoConfig
from xmixup.tokenization.auto import AutoTokenizer
from xmixup.tasks.auto_model import AutoModelForTask
from xmixup.trainers.trainer_qa import QATrainer as Trainer
from xmixup.trainers.training_args import TrainingArguments
from xmixup.utils.hf_argparser import HfArgumentParser
from xmixup.data.tasks.qa import squad_convert_examples_to_features
from xmixup.data.datasets.qa import SquadProcessor
from xmixup.metrics.squad import compute_predictions_logits, squad_evaluate


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

    data_dir: Optional[str] = field(
        metadata={"help": "The input data dir. Should contain the .json files for the SQuAD task."}
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
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": "The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    do_lower_case: bool = field(
        default=True, metadata={"help": "If true, do lower case."}
    )
    verbose_logging: bool = field(
        default=False, metadata={"help": "If true, all of the warnings related to data processing will be printed. "
                                         "A number of warnings are expected for a normal SQuAD evaluation."}
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, the SQuAD examples contain some that do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    n_best_size: int = field(
        default=20, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    threads: int = field(
        default=1, metadata={"help": "Multiple threads for converting example to features."}
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

    # Prepare Question-Answering task
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    train_languages = data_args.train_languages.split(',')
    eval_languages = data_args.eval_languages.split(',')
    processor = SquadProcessor(languages=eval_languages, train_languages=train_languages, task_name=data_args.task_name)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    config.n_langs = len(processor.get_languages())
    lang2id = processor.get_lang2id()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    with training_args.strategy.scope():
        model = AutoModelForTask.from_pretrained(
            "qa",
            model_args.model_name_or_path,
            from_pt=True,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Get datasets
    train_examples = processor.get_train_examples(data_args.data_dir) if training_args.do_train else None
    en_train_examples = processor.get_train_examples(data_args.data_dir, lang="en") if training_args.do_train else None
    back_trans_train_examples = processor.get_train_examples(data_args.data_dir, add_back_trans=True) \
        if training_args.do_train else None

    train_dataset = (
        squad_convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            doc_stride=data_args.doc_stride,
            max_query_length=data_args.max_query_length,
            is_training=True,
            threads=data_args.threads,
            lang2id=lang2id,
            en_examples=en_train_examples,
            trans_examples=back_trans_train_examples
        )
        if training_args.do_train
        else None
    )

    eval_examples = []
    eval_features = []
    eval_datasets = []
    eval_en_examples = processor.get_dev_examples(data_args.data_dir, lang="en") if training_args.do_eval else None
    if "xquad" in data_args.task_name:
        dev_languages = ["ar", "de", "en", "es", "hi", "vi", "zh"]
    elif "tydiqa" in data_args.task_name:
        dev_languages = ["en"]
    else:
        dev_languages = eval_languages
    for lang in dev_languages:
        eval_example = processor.get_dev_examples(data_args.data_dir, lang=lang) if training_args.do_eval else None
        eval_feature, eval_dataset = (
            squad_convert_examples_to_features(
                examples=eval_example,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                doc_stride=data_args.doc_stride,
                max_query_length=data_args.max_query_length,
                lang2id=lang2id,
                threads=data_args.threads,
                en_examples=eval_en_examples
            )
            if training_args.do_eval
            else (None, None)
        )
        eval_examples.append(eval_example)
        eval_features.append(eval_feature)
        eval_datasets.append(eval_dataset)

    training_data_args = {"n_best_size": data_args.n_best_size, "max_answer_length": data_args.max_answer_length,
                          "verbose_logging": data_args.verbose_logging, "do_lower_case": data_args.do_lower_case,
                          "version_2_with_negative": data_args.version_2_with_negative,
                          "null_score_diff_threshold": data_args.null_score_diff_threshold, "tokenizer": tokenizer
                          }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_args=training_data_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        eval_languages=dev_languages,
        eval_examples=eval_examples,
        eval_features=eval_features,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    # Predict
    if training_args.do_predict:
        f1_results = []
        em_results = []
        _, test_en_examples = processor.get_test_examples_by_lang(data_args.data_dir, "en")
        for lang in eval_languages:
            input_data, test_examples = processor.get_test_examples_by_lang(data_args.data_dir, lang)
            _, test_trans_examples = processor.get_test_trans_examples_by_lang(data_args.data_dir, lang)
            test_features, test_dataset = (
                squad_convert_examples_to_features(
                    examples=test_examples,
                    tokenizer=tokenizer,
                    max_seq_length=data_args.max_seq_length,
                    doc_stride=data_args.doc_stride,
                    max_query_length=data_args.max_query_length,
                    lang2id=lang2id,
                    threads=data_args.threads,
                    en_examples=test_en_examples,
                    trans_examples=test_trans_examples
                )
                if training_args.do_predict
                else (None, None)
            )
            # Compute predictions
            pred_dir = os.path.join(training_args.output_dir, "predictions")
            output_prediction_file = os.path.join(pred_dir, "{}_predictions.json".format(lang))
            output_nbest_file = os.path.join(pred_dir, "{}_nbest_predictions.json".format(lang))

            if data_args.version_2_with_negative:
                output_null_log_odds_file = os.path.join(pred_dir, "{}_null_odds.json".format(lang))
            else:
                output_null_log_odds_file = None

            all_results = trainer.predict(test_dataset, test_features=test_features)
            predictions = compute_predictions_logits(
                test_examples, test_features, all_results,
                n_best_size=data_args.n_best_size,
                max_answer_length=data_args.max_answer_length,
                do_lower_case=data_args.do_lower_case,
                verbose_logging=data_args.verbose_logging,
                version_2_with_negative=data_args.version_2_with_negative,
                null_score_diff_threshold=data_args.null_score_diff_threshold,
                tokenizer=tokenizer,
                output_prediction_file=output_prediction_file,
                output_nbest_file=output_nbest_file,
                output_null_log_odds_file=output_null_log_odds_file,
                is_predict=True)
            if "mlqa" in data_args.task_name:
                from xmixup.metrics.mlqa import evaluate
                results = evaluate(input_data, predictions, lang)
            else:
                results = squad_evaluate(test_examples, predictions)
            f1 = results["f1"]
            exact_match = results["exact_match"]
            logger.info("\nLanguage {} result: f1: {}, EM: {}".format(lang, f1, exact_match))
            output_test_results_file = os.path.join(training_args.output_dir, lang + "_test_results.txt")
            with tf.io.gfile.GFile(output_test_results_file, "w") as writer:
                writer.write("Language %s: %s\n" % (lang, results))
            f1_results.append(f1)
            em_results.append(exact_match)

        mean_f1 = tf.math.reduce_mean(f1_results)
        mean_em = tf.math.reduce_mean(em_results)
        logger.info("Test mean results: f1: {}, EM: {}".format(mean_f1, mean_em))
        output_test_results_file = os.path.join(training_args.output_dir, "ave_test_results.txt")
        with tf.io.gfile.GFile(output_test_results_file, "w") as writer:
            writer.write("Test mean results: f1: {}, EM: {}".format(mean_f1, mean_em))


if __name__ == "__main__":
    main()
