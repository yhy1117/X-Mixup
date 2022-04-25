""" PAWS-X utils"""

import logging
import os
import tensorflow as tf

from xmixup.data.datasets.base import DataProcessor
from xmixup.data.tasks.classification import InputExample


logger = logging.getLogger(__name__)


class Processor(DataProcessor):
    """Processor for the PAWS-X dataset."""

    def __init__(self, task_name, languages, train_languages=None):
        self.task_name = task_name
        self.languages = languages
        self.train_languages = train_languages

    def get_train_examples(self, data_dir, add_trans=False, add_back_trans=False):
        """See base class."""
        examples = []
        if add_trans:
            en_lines = self._read_tsv(os.path.join(os.path.join(data_dir, "en"), "train.tsv"))
        for lg in self.train_languages:
            lines = self._read_tsv(os.path.join(os.path.join(data_dir, lg), "train.tsv"))
            if add_back_trans and lg != "en":
                back_trans_lines = self._read_tsv(os.path.join(data_dir, f"translate-train-en/translated_train.{lg}-en.tsv"))
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % ("train", i)
                text_a = line[1]
                text_b = line[2]
                label = str(line[3])
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg)
                if add_trans:
                    label_en = str(en_lines[i][3])
                    assert label == label_en
                    example.text_a_en = en_lines[i][1]
                    example.text_b_en = en_lines[i][2]
                if add_back_trans:
                    if lg != "en":
                        example.text_a_back_trans = back_trans_lines[i][1]
                        example.text_b_back_trans = back_trans_lines[i][2]
                    else:
                        example.text_a_back_trans = text_a
                        example.text_b_back_trans = text_b
                examples.append(example)

        return examples

    def get_dev_examples(self, data_dir, add_trans=False):
        """See base class."""
        examples = []
        for lg in self.languages:
            lines = self._read_tsv(os.path.join(os.path.join(data_dir, lg), "dev_2k.tsv"))
            if add_trans:
                en_lines = self._read_tsv(os.path.join(os.path.join(data_dir, "translate-dev"), f"dev_2k.{lg}.tsv"))
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % ("test", i)
                id = line[0]
                text_a = line[1]
                text_b = line[2]
                label = str(line[3])
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg)
                if add_trans:
                    id_en = en_lines[i][0]
                    assert id == id_en
                    label_en = str(en_lines[i][3])
                    if label != label_en:
                        example.label = label_en
                    example.text_a_en = en_lines[i][1]
                    example.text_b_en = en_lines[i][2]
                examples.append(example)

        return examples

    def get_test_examples_by_lang(self, data_dir, lang, add_trans=False):
        """See base class."""
        lines = self._read_tsv(os.path.join(os.path.join(data_dir, lang), "test_2k.tsv"))
        if add_trans and lang != "en":
            trans_lines = self._read_tsv(os.path.join(os.path.join(data_dir, "translate-test"), f"test.{lang}.tsv"))
            assert len(lines) - 1 == len(trans_lines)
        examples = []
        for (i, line) in enumerate(lines[1:]):
            guid = "%s-%s" % ("test", i)
            text_a = line[1]
            text_b = line[2]
            label = str(line[3])
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lang)
            if add_trans:
                if lang != "en":
                    trans_line = trans_lines[i]
                    trans_label = str(trans_line[2])
                    assert label == trans_label
                    example.text_a_en = trans_line[0]
                    example.text_b_en = trans_line[1]
                else:
                    example.text_a_en = text_a
                    example.text_b_en = text_b
                examples.append(example)

        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_languages(self):
        """Get languages for dataset"""

        return self.train_languages

    def save_predictions(self, output_file, data_dir, lang, pred_ids):
        """Save predictions for test data"""
        with tf.io.gfile.GFile(output_file, "w") as writer:
            lines = self._read_tsv(os.path.join(os.path.join(data_dir, lang), "test_2k.tsv"))
            example_id = 0
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                output_line = line[1] + "\t" + line[2] + '\t' + line[3] + '\t' + str(pred_ids[example_id]) + "\n"
                writer.write(output_line)
                example_id += 1
