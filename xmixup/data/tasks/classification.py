"""Data utils for sequence classification"""
import json
import dataclasses
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
import tensorflow as tf

from xmixup.data.datasets.base import DataProcessor
from xmixup.tokenization.utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        language: (Optional) string. The language of the example.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    language: Optional[str] = None
    text_a_en: Optional[str] = None
    text_b_en: Optional[str] = None
    text_a_back_trans: Optional[str] = None
    text_b_back_trans: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        langs: (Optional) Language corresponding to the input.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    langs: Optional[List[int]] = None
    input_ids_en: Optional[List[int]] = None
    attention_mask_en: Optional[List[int]] = None
    token_type_ids_en: Optional[List[int]] = None
    input_ids_back_trans: Optional[List[int]] = None
    attention_mask_back_trans: Optional[List[int]] = None
    token_type_ids_back_trans: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    processor: DataProcessor,
    max_length: Optional[int] = None,
    label_list=None,
    output_mode="classification",
    is_training=False,
    mixup_inference=False,
    add_back_trans=False
):
    """Convert examples to features"""
    if max_length is None:
        max_length = tokenizer.max_len

    if label_list is None:
        raise ValueError("Please provide label list.")

    language_list = processor.get_languages()
    language_map = {lang: i for i, lang in enumerate(language_list)}
    label_map = {label: i for i, label in enumerate(label_list)}

    def language_from_example(example: InputExample) -> Union[int, None]:
        if example.language is None:
            return None
        else:
            return language_map[example.language]

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    languages = [language_from_example(example) for example in examples]
    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    if is_training or mixup_inference:
        batch_encoding_en = tokenizer(
            [(example.text_a_en, example.text_b_en) for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
    if is_training and add_back_trans:
        batch_encoding_back_trans = tokenizer(
            [(example.text_a_back_trans, example.text_b_back_trans) for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        langs = [languages[i]]
        feature = InputFeatures(**inputs, label=labels[i], langs=langs)
        if is_training or mixup_inference:
            feature.input_ids_en = batch_encoding_en["input_ids"][i]
            feature.attention_mask_en = batch_encoding_en["attention_mask"][i]
            if "token_type_ids" in batch_encoding_en:
                feature.token_type_ids_en = batch_encoding_en["token_type_ids"][i]
        if is_training:
            if add_back_trans:
                feature.input_ids_back_trans = batch_encoding_back_trans["input_ids"][i]
                feature.attention_mask_back_trans = batch_encoding_back_trans["attention_mask"][i]
                if "token_type_ids" in batch_encoding_back_trans:
                    feature.token_type_ids_back_trans = batch_encoding_back_trans["token_type_ids"][i]

        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("language: %s" % (example.language))
        logger.info("label: %s" % (example.label))
        logger.info("features: %s" % features[i])

    """convert features to tf.data.Dataset"""
    def gen():
        for ex in features:
            d = {k: v for k, v in asdict(ex).items() if v is not None}
            label = d.pop("label")
            yield (d, label)

    input_names_base = ["input_ids"] + tokenizer.model_input_names
    input_names = input_names_base + ["langs"]
    if is_training or mixup_inference:
        input_names += [k + "_en" for k in input_names_base]
        if is_training and add_back_trans:
            input_names += [k + "_back_trans" for k in input_names_base]

    feature_types = {k: tf.int32 for k in input_names}
    feature_shapes = {k: tf.TensorShape([None]) for k in input_names}
    return tf.data.Dataset.from_generator(
        gen,
        (feature_types, tf.int64),
        (feature_shapes, tf.TensorShape([])),
    )
