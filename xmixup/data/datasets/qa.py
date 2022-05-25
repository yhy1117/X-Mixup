""" SQuAD utils"""

import os
import json

from xmixup.data.datasets.base import DataProcessor
from xmixup.data.tasks.qa import SquadExample


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """
    def __init__(self, task_name, languages, train_languages=None):
        self.task_name = task_name
        self.languages = languages
        self.train_languages = train_languages

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in dataset:
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, lang=None, add_back_trans=False):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one.

        """
        examples = []
        if not add_back_trans:
            if "xquad" in self.task_name or "mlqa" in self.task_name:
                if "en" in self.train_languages and len(self.train_languages) == 1 or lang == "en":
                    data_dir = os.path.join(data_dir, "squad")
                    filename = "train-v1.1.json"
                    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                        input_data = json.load(reader)["data"]
                    examples = self._create_examples(input_data, "train", "en")
                else:
                    for lang in self.train_languages:
                        if lang == "en":
                            data_dir = os.path.join(data_dir, "squad")
                            filename = "train-v1.1.json"
                        else:
                            data_dir = os.path.join(data_dir, "squad/translate-train")
                            filename = f"squad.translate.train.en-{lang}.json"
                        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                            input_data = json.load(reader)["data"]
                            example = self._create_examples(input_data, "train", lang)
                            examples += example

            elif "tydiqa" in self.task_name:
                data_dir = os.path.join(data_dir, "tydiqa/translate-train")
                for lang in self.train_languages:
                    filename = "tydiqa.translate.train.en-{}.json".format(lang)
                    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                        input_data = json.load(reader)["data"]
                        example = self._create_examples(input_data, "train", lang)
                        examples += example
        else:
            if "xquad" in self.task_name or "mlqa" in self.task_name:
                data_dir = os.path.join(data_dir, "squad/translate-train-en")
                for lang in self.train_languages:
                    filename = f"squad.translate.train.en-{lang}-en.json"
                    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                        input_data = json.load(reader)["data"]
                        example = self._create_examples(input_data, "train", lang, back_trans=True)
                        examples += example
            else:
                data_dir = os.path.join(data_dir, "tydiqa/translate-train-en")
                for lang in self.train_languages:
                    filename = "tydiqa.translate.train.en-{}-en.json".format(lang)
                    with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                        input_data = json.load(reader)["data"]
                        example = self._create_examples(input_data, "train", lang, back_trans=True)
                        examples += example

        return examples

    def get_dev_examples(self, data_dir, filename=None, lang=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one.
        """
        examples = []
        if "xquad" in self.task_name or "mlqa" in self.task_name:
            data_dir = os.path.join(data_dir, "mlqa/dev")
        elif "tydiqa" in self.task_name:
            data_dir = os.path.join(data_dir, "tydiqa/dev")
        if lang is not None:
            eval_langs = [lang]
        else:
            eval_langs = self.languages
        for lang in eval_langs:
            if "xquad" in self.task_name or "mlqa" in self.task_name:
                filename = "dev-context-{}-question-{}.json".format(lang, lang)
            elif "tydiqa" in self.task_name:
                filename = "tydiqa.{}.dev.json".format(lang)
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)["data"]
                example = self._create_examples(input_data, "dev", lang)
                examples += example

        return examples

    def get_test_examples_by_lang(self, data_dir, lang):
        """
        Returns the testing examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for testing.
            lang: None by default, the testing language.
        """
        if "xquad" in self.task_name:
            data_dir = os.path.join(data_dir, "xquad/test")
            filename = "xquad.{}.json".format(lang)
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)["data"]
                examples = self._create_examples(input_data, "test", lang)
        elif "mlqa" in self.task_name:
            data_dir = os.path.join(data_dir, "mlqa/test")
            filename = "test-context-{}-question-{}.json".format(lang, lang)
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)["data"]
                examples = self._create_examples(input_data, "test", lang)
        elif "tydiqa" in self.task_name:
            data_dir = os.path.join(data_dir, "tydiqa/dev")
            filename = "tydiqa.{}.dev.json".format(lang)
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)["data"]
                examples = self._create_examples(input_data, "test", lang)

        return input_data, examples

    def get_test_trans_examples_by_lang(self, data_dir, lang):
        """
        Returns the testing examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for testing.
            lang: None by default, the testing language.
        """
        if "xquad" in self.task_name:
            data_dir = os.path.join(data_dir, "xquad/translate-test")
            filename = "xquad.translate.test.{}-en.json".format(lang)
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)["data"]
                examples = self._create_examples(input_data, "test", lang)
        elif "mlqa" in self.task_name:
            data_dir = os.path.join(data_dir, "mlqa/translate-test")
            filename = "mlqa.translate.test.{}-en.json".format(lang)
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)["data"]
                examples = self._create_examples(input_data, "test", lang)
        elif "tydiqa" in self.task_name:
            data_dir = os.path.join(data_dir, "tydiqa/translate-test")
            filename = "tydiqa.translate.test.{}-en.json".format(lang)
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
                input_data = json.load(reader)["data"]
                examples = self._create_examples(input_data, "test", lang)

        return input_data, examples

    def _create_examples(self, input_data, set_type, language, back_trans=False):
        is_training = set_type == "train"
        examples = []
        for entry in input_data:
            title = entry["title"] if "title" in entry else ""
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    if back_trans:
                        qas_id += f"_{language}"
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            start_position_character = answer["answer_start"]
                            if start_position_character == -1 or start_position_character >= len(context_text):
                                continue
                            answer_text = answer["text"]
                        else:
                            answers = qa["answers"]

                        example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            context_text=context_text,
                            answer_text=answer_text,
                            start_position_character=start_position_character,
                            title=title,
                            is_impossible=is_impossible,
                            answers=answers,
                            language=language
                        )

                        examples.append(example)
        return examples

    def get_languages(self):
        """Get languages for dataset"""

        return self.train_languages

    def get_lang2id(self):
        langs = self.get_languages()
        lang2id = {lang: i for i, lang in enumerate(langs)}

        return lang2id
