""" XNLI utils """
import logging
import os
import tensorflow as tf

from xmixup.data.datasets.base import DataProcessor
from xmixup.data.tasks.classification import InputExample


logger = logging.getLogger(__name__)


class Processor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self, task_name, languages, train_languages=None):
        self.task_name = task_name
        self.languages = languages
        self.train_languages = train_languages

    def get_train_examples(self, data_dir, add_trans=False, add_back_trans=False):
        """See base class."""
        examples = []
        if add_trans:
            en_lines = self._read_tsv(os.path.join(data_dir, "XNLI-MT-1.0/multinli/multinli.train.en.tsv"))
        for lg in self.train_languages:
            lines = self._read_tsv(os.path.join(data_dir, "XNLI-MT-1.0/multinli/multinli.train.{}.tsv".format(lg)))
            if add_back_trans:
                back_trans_lines = self._read_tsv(os.path.join(data_dir, f"translate-train-en/multinli.train.{lg}-en.tsv"))
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = "%s-%s" % ("train", i)
                text_a = line[0]
                text_b = line[1]
                label = "contradiction" if line[2] == "contradictory" else line[2]
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg)
                if add_trans:
                    label_en = "contradiction" if en_lines[i][2] == "contradictory" else en_lines[i][2]
                    assert label == label_en
                    example.text_a_en = en_lines[i][0]
                    example.text_b_en = en_lines[i][1]
                if add_back_trans:
                    example.text_a_back_trans = back_trans_lines[i][0]
                    example.text_b_back_trans = back_trans_lines[i][1]
                examples.append(example)

        return examples

    def get_dev_examples(self, data_dir, add_trans=False):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.dev.tsv"))
        examples = []
        if add_trans:
            trans_lines = self._read_tsv(os.path.join(os.path.join(data_dir, "translate-dev"), f"xnli.dev.tsv"))
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            lang = line[0]
            if lang not in self.languages:
                continue
            guid = "%s-%s" % ("dev", i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lang)
            if add_trans:
                en_line = trans_lines[i]
                text_a_en = en_line[6]
                text_b_en = en_line[7]
                label_en = en_line[1]
                assert label == label_en
                example.text_a_en = text_a_en
                example.text_b_en = text_b_en
            examples.append(example)

        return examples

    def get_test_examples_by_lang(self, data_dir, lang, add_trans=False):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
        if add_trans and lang != "en":
            trans_lines = self._read_tsv(os.path.join(os.path.join(data_dir, "translate-test"), f"test-{lang}-en-translated.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            language = line[0]
            if language != lang:
                continue
            guid = "%s-%s" % ("test", i)
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lang)
            if add_trans:
                if lang != "en":
                    trans_line = trans_lines.pop(0)
                    text_a_trans = trans_line[0]
                    text_b_trans = trans_line[1]
                    trans_label = trans_line[2]
                    assert label == trans_label
                else:
                    text_a_trans = text_a
                    text_b_trans = text_b
                example.text_a_en = text_a_trans
                example.text_b_en = text_b_trans
            examples.append(example)

        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_languages(self):
        """Get languages for dataset"""

        return self.train_languages

    def save_predictions(self, output_file, data_dir, lang, pred_ids):
        """Save predictions for test data"""
        if "_" not in self.task_name:
            data_dir = os.path.join(data_dir, "XNLI-1.0")
        with tf.io.gfile.GFile(output_file, "w") as writer:
            with tf.io.gfile.GFile(os.path.join(data_dir, "xnli.test.tsv"), "r") as f:
                example_id = 0
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('\t')
                    if line[0] != lang:
                        continue
                    output_line = line[6] + "\t" + line[7] + '\t' + line[1] + '\t' + str(pred_ids[example_id]) + "\n"
                    writer.write(output_line)
                    example_id += 1
