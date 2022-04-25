"""Data utils for tagging"""
import dataclasses
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional
import tensorflow as tf
from xmixup.data.datasets.base import DataProcessor
from xmixup.tokenization.utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for simple tagging.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: list. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        language: (Optional) string. The language of the example.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]] = None
    language: Optional[str] = None
    trans_words: Optional[List[str]] = None
    back_trans_words: Optional[List[str]] = None
    soft_labels: Optional[List[str]] = None

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
        label_ids: (Optional) Label corresponding to the input.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    pooling_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    langs: Optional[List[int]] = None
    input_ids_en: Optional[List[int]] = None
    attention_mask_en: Optional[List[int]] = None
    soft_pooling_ids: Optional[List[int]] = None
    soft_label_ids: Optional[List[List[float]]] = None
    input_ids_back_trans: Optional[List[int]] = None
    attention_mask_back_trans: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    processor: DataProcessor,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    pad_token_label_id=-100,
    is_training=False,
    add_trans=False,
    add_back_trans=False
):
    """Convert examples to features"""
    if max_length is None:
        max_length = tokenizer.max_len

    if label_list is None:
        raise ValueError("Please provide label list.")

    if task is not None:
        logger.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    language_list = processor.get_languages()
    language_map = {lang: i for i, lang in enumerate(language_list)}

    features = []
    pad_soft_label = [0.0] * len(label_list)
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        trans_tokens = []
        back_trans_tokens = []
        label_ids = []
        soft_label_ids = []
        pooling_ids = []
        soft_pooling_ids = []
        if is_training:
            words = example.trans_words
        else:
            words = example.words
        for word, label in zip(words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word) != 0 and len(word_tokens) == 0:
                word_tokens = [tokenizer.unk_token]
            if is_training:
                trans_tokens.extend(word_tokens)
            else:
                tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            pooling_ids.extend([len(pooling_ids) + 1] * len(word_tokens))

        if is_training:
            words = example.words
        else:
            words = example.trans_words
        for i, word in enumerate(words):
            word_tokens = tokenizer.tokenize(word)
            if len(word) != 0 and len(word_tokens) == 0:
                word_tokens = [tokenizer.unk_token]
            if is_training:
                tokens.extend(word_tokens)
                soft_label = example.soft_labels[i].split(';')
                soft_label = [float(label) for label in soft_label]
                soft_label_ids.extend([soft_label] + [pad_soft_label] * (len(word_tokens) - 1))
                soft_pooling_ids.extend([len(soft_pooling_ids) + 1] * len(word_tokens))
            else:
                trans_tokens.extend(word_tokens)

        if is_training and add_back_trans:
            for word in example.back_trans_words:
                word_tokens = tokenizer.tokenize(word)
                if len(word) != 0 and len(word_tokens) == 0:
                    word_tokens = [tokenizer.unk_token]
                back_trans_tokens.extend(word_tokens)

        # padding to max length
        if is_training:
            input_ids, attention_mask, soft_label_ids, soft_pooling_ids = convert_example_to_feature(tokenizer, tokens,
                                                                                                      max_length,
                                                                                                      label_ids=soft_label_ids,
                                                                                                      pooling_ids=soft_pooling_ids,
                                                                                                      pad_token_label_id=pad_soft_label)
            input_ids_en, attention_mask_en, label_ids, pooling_ids = convert_example_to_feature(tokenizer,
                                                                                                 trans_tokens,
                                                                                                 max_length,
                                                                                                 label_ids=label_ids,
                                                                                                 pooling_ids=pooling_ids,
                                                                                                 pad_token_label_id=-100)
            if add_back_trans:
                input_ids_back_trans, attention_mask_back_trans, _, _ = convert_example_to_feature(tokenizer,
                                                                                                   back_trans_tokens,
                                                                                                   max_length)
        else:
            input_ids, attention_mask, label_ids, pooling_ids = convert_example_to_feature(tokenizer, tokens, max_length,
                                                                                           label_ids, pooling_ids,
                                                                                           pad_token_label_id=-100)
            input_ids_en, attention_mask_en, _, _ = convert_example_to_feature(tokenizer, trans_tokens, max_length)

        if example.language and len(example.language) > 0:
            langs = [language_map[example.language]]
        else:
            langs = None

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(label_ids) == max_length
        assert len(pooling_ids) == max_length
        if add_trans:
            assert len(input_ids_en) == max_length
            assert len(attention_mask_en) == max_length
            if is_training:
                assert len(soft_label_ids) == max_length
                assert len(soft_pooling_ids) == max_length

        feature = InputFeatures(input_ids=input_ids, attention_mask=attention_mask, label_ids=label_ids, langs=langs,
                                pooling_ids=pooling_ids)
        if add_trans:
            feature.input_ids_en = input_ids_en
            feature.attention_mask_en = attention_mask_en
            if is_training:
                feature.soft_label_ids = soft_label_ids
                feature.soft_pooling_ids = soft_pooling_ids
                if add_back_trans:
                    feature.input_ids_back_trans = input_ids_back_trans
                    feature.attention_mask_back_trans = attention_mask_back_trans

        features.append(feature)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("langs: {}".format(langs))
            logger.info("features: %s" % feature)

    """convert features to tf.data.Dataset"""
    def gen():
        for ex in features:
            d = {k: v for k, v in asdict(ex).items() if v is not None}
            label = d.pop("label_ids")
            yield (d, label)

    input_names = ["input_ids", "attention_mask", "pooling_ids", "langs"]
    if add_trans:
        input_names += ["input_ids_en", "attention_mask_en"]
        if is_training:
            input_names += ["soft_pooling_ids"]
    if is_training and add_back_trans:
        input_names += ["input_ids_back_trans", "attention_mask_back_trans"]

    feature_types = {k: tf.int32 for k in input_names}
    feature_shapes = {k: tf.TensorShape([None]) for k in input_names}
    if is_training and add_trans:
        soft_label_type = {"soft_label_ids": tf.float32}
        feature_types.update(soft_label_type)
        soft_label_shape = {"soft_label_ids": tf.TensorShape([None, len(label_list)])}
        feature_shapes.update(soft_label_shape)
    return tf.data.Dataset.from_generator(
        gen,
        (feature_types, tf.int64),
        (feature_shapes, tf.TensorShape([None])),
    )


def convert_example_to_feature(tokenizer, tokens, max_length, label_ids=None, pooling_ids=None, sep_token_extra=True, pad_token_label_id=None):
    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    if isinstance(pad_token_label_id, (float, int, list)):
        pad_token_label_id = [pad_token_label_id]
    special_tokens_count = 3 if sep_token_extra else 2
    if len(tokens) > max_length - special_tokens_count:
        tokens = tokens[:(max_length - special_tokens_count)]
        if label_ids is not None:
            label_ids = label_ids[:(max_length - special_tokens_count)]
        if pooling_ids is not None:
            pooling_ids = pooling_ids[:(max_length - special_tokens_count)]
    tokens += [tokenizer.sep_token]
    if label_ids is not None:
        label_ids += pad_token_label_id
    if pooling_ids is not None:
        pooling_ids += [0]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [tokenizer.sep_token]
        if label_ids is not None:
            label_ids += pad_token_label_id
        if pooling_ids is not None:
            pooling_ids += [0]

    tokens = [tokenizer.cls_token] + tokens
    if label_ids is not None:
        label_ids = pad_token_label_id + label_ids
    if pooling_ids is not None:
        pooling_ids = [0] + pooling_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    input_ids += ([pad_token] * padding_length)
    input_mask += ([0] * padding_length)
    if label_ids is not None:
        label_ids += (pad_token_label_id * padding_length)
    if pooling_ids is not None:
        pooling_ids += ([0] * padding_length)

    return input_ids, input_mask, label_ids, pooling_ids
