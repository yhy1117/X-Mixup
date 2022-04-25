"""Tagging utils"""
import os
import tensorflow as tf

from xmixup.data.datasets.base import DataProcessor
from xmixup.data.tasks.ner import InputExample


class TagProcessor(DataProcessor):
    """
    Processor for the tagging tasks, eg. NER, Pos tagging.
    """
    def __init__(self, task_name, languages, train_languages=None):
        self.task_name = task_name
        self.languages = languages
        self.train_languages = train_languages

    def get_examples(self, data_dir, set_type="train", lang=None):
        examples = []
        if lang is None:
            if set_type == "train":
                langs = self.train_languages
            else:
                langs = self.languages
        else:
            langs = [lang]
        for lang in langs:
            filename = os.path.join(data_dir, os.path.join(lang, "{}.tsv".format(set_type)))
            if not tf.io.gfile.exists(filename): continue
            lines = self._read_tsv(filename)
            idx = 0
            words = []
            labels = []
            for line in lines:
                if len(line) < 2:
                    guid = "%s-%d" % (lang, idx)
                    examples.append(InputExample(guid=guid, words=words, labels=labels, language=lang))
                    idx += 1
                    words = []
                    labels = []
                    continue
                words.append(line[0])
                labels.append(line[1])

        return examples

    def read_tagging_file(self, filename):
        lines = self._read_tsv(filename)
        words_list, labels_list = [], []
        words, labels = [], []
        for line in lines:
            if len(line) < 2:
                words_list.append(words)
                labels_list.append(labels)
                words = []
                labels = []
            else:
                words.append(line[0])
                labels.append(line[1])

        return words_list, labels_list

    def get_examples_with_trans(self, data_dir, set_type="train", lang=None, add_back_trans=False):
        examples = []
        en_words_list, labels_list, soft_labels, back_trans_words_list = None, None, None, None
        if lang is None:
            if set_type == "train":
                langs = self.train_languages
            else:
                langs = self.languages
        else:
            langs = [lang]
        if set_type == "train":
            en_words_list, labels_list = self.read_tagging_file(os.path.join(data_dir, os.path.join("en", "train.tsv")))
        for lang in langs:
            if set_type == "train":
                filename = os.path.join(data_dir, os.path.join("translate-train-logits", f"{lang}.txt"))
                if not tf.io.gfile.exists(filename):
                    continue
                words_list, soft_labels = self.read_tagging_file(filename)
                if add_back_trans:
                    back_trans_file = os.path.join(data_dir, os.path.join("translate-train-en-token", f"{lang}.txt"))
                    back_trans_words_list, _ = self.read_tagging_file(back_trans_file)
            else:
                filename = os.path.join(data_dir, os.path.join(lang, f"{set_type}.tsv"))
                if not tf.io.gfile.exists(filename):
                    continue
                words_list, labels_list = self.read_tagging_file(filename)
                en_words_list, _ = self.read_tagging_file(
                    os.path.join(data_dir, os.path.join(f"translate-{set_type}-token", f"{lang}.txt")))
            assert len(words_list) == len(en_words_list)
            if add_back_trans:
                assert len(words_list) == len(back_trans_words_list)
            if set_type == "dev":
                if len(words_list) > 1000:
                    words_list = words_list[:1000]
                en_words_list = en_words_list[:len(words_list)]
            for idx in range(len(words_list)):
                guid = "%s-%d" % (lang, idx)
                example = InputExample(guid=guid, words=words_list[idx], labels=labels_list[idx], language=lang)
                example.trans_words = en_words_list[idx]
                if set_type == "train":
                    example.soft_labels = soft_labels[idx]
                    if add_back_trans:
                        example.back_trans_words = back_trans_words_list[idx]

                examples.append(example)

        return examples

    def get_labels_from_file(self, data_dir, label_file):
        """Get labels from file"""
        with tf.io.gfile.GFile(os.path.join(data_dir, label_file), "r") as f:
            lines = f.readlines()
            labels = []
            for line in lines:
                if len(line) > 1:
                    label = line.strip()
                    assert isinstance(label, str)
                    labels.append(label)

        return labels

    def get_languages(self):
        """Get languages for dataset"""

        return self.train_languages

    def save_predictions(self, output_file, data_dir, lang, pred_ids):
        """Save predictions for test data"""
        with tf.io.gfile.GFile(output_file, "w") as writer:
            with tf.io.gfile.GFile(os.path.join(data_dir, os.path.join(lang, "test.tsv")), "r") as f:
                example_id = 0
                lines = f.readlines()
                for line in lines:
                    if line == "" or line == "\n":
                        example_id += 1
                        writer.write("\n")
                    else:
                        token = line.split("\t")[0]
                        if len(pred_ids[example_id]) > 0:
                            output_line = token + "\t" + pred_ids[example_id].pop(0) + "\n"
                        else:
                            output_line = token + "\t" + "O" + "\n"
                        writer.write(output_line)
