import numpy as np
import functools
import tensorflow as tf

from xmixup.configuration.utils import PretrainedConfig


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
    Args:
        initializer_range: float, initializer range for stddev.
    Returns:
        TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = tf.constant(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = tf.constant(np.cos(position_enc[:, 1::2]))


def get_masks(slen, lengths, causal, padding_mask=None, dtype=tf.float32):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    bs = shape_list(lengths)[0]
    if padding_mask is not None:
        mask = padding_mask
    else:
        # assert lengths.max().item() <= slen
        alen = tf.range(slen)
        mask = tf.math.less(alen, lengths[:, tf.newaxis])

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = tf.less_equal(
            tf.tile(alen[tf.newaxis, tf.newaxis, :], (bs, slen, 1)), alen[tf.newaxis, :, tf.newaxis]
        )
    else:
        attn_mask = mask

    # sanity check
    # assert shape_list(mask) == [bs, slen]
    tf.debugging.assert_equal(shape_list(mask), [bs, slen])
    assert causal is False or shape_list(attn_mask) == [bs, slen, slen]

    mask = tf.cast(mask, dtype=dtype)
    attn_mask = tf.cast(attn_mask, dtype=dtype)

    return mask, attn_mask


def cast_bool_to_primitive(bool_variable, default_tensor_to_true=False):
    """Function arguments can be inserted as boolean tensor
        and bool variables to cope with keras serialization
        we need to cast `output_attentions` to correct bool
        if it is a tensor

    Args:
        default_tensor_to_true: bool, if tensor should default to True
        in case tensor has no numpy attribute
    """
    # if bool variable is tensor and has numpy value
    if tf.is_tensor(bool_variable):
        if hasattr(bool_variable, "numpy"):
            return bool(bool_variable.numpy())
        elif default_tensor_to_true:
            return True

    # else variable is bool
    return bool_variable


def keras_serializable(cls):
    """
    Decorate a Keras Layer class to support Keras serialization.

    This is done by:
    1. adding a `transformers_config` dict to the Keras config dictionary in `get_config` (called by Keras at
       serialization time
    2. wrapping `__init__` to accept that `transformers_config` dict (passed by Keras at deserialization time) and
       convert it to a config object for the actual layer initializer
    3. registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does
       not need to be supplied in `custom_objects` in the call to `tf.keras.models.load_model`

    :param cls: a tf.keras.layers.Layers subclass that accepts a `config` argument to its initializer (typically a
                `TF*MainLayer` class in this project)
    :return: the same class object, with modifications for Keras deserialization.
    """
    initializer = cls.__init__

    config_class = getattr(cls, "config_class", None)
    if config_class is None:
        raise AttributeError("Must set `config_class` to use @keras_serializable")

    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        transformers_config = kwargs.pop("transformers_config", None)
        config = args[0] if args and isinstance(args[0], PretrainedConfig) else kwargs.get("config", None)
        if config is not None and transformers_config is not None:
            raise ValueError("Must pass either `config` or `transformers_config`, not both")
        elif config is not None:
            # normal layer construction, call with unchanged args (config is already in there)
            initializer(self, *args, **kwargs)
        elif transformers_config is not None:
            # Keras deserialization, convert dict to config
            config = config_class.from_dict(transformers_config)
            initializer(self, config, *args, **kwargs)
        else:
            raise ValueError("Must pass either `config` (PretrainedConfig) or `transformers_config` (dict)")
        self._transformers_config = config
        self._kwargs = kwargs

    cls.__init__ = wrapped_init

    if not hasattr(cls, "get_config"):
        raise TypeError("Only use @keras_serializable on tf.keras.layers.Layer subclasses")
    if hasattr(cls.get_config, "_is_default"):

        def get_config(self):
            cfg = super(cls, self).get_config()
            cfg["transformers_config"] = self._transformers_config.to_dict()
            cfg.update(self._kwargs)
            return cfg

        cls.get_config = get_config

    cls._keras_serializable = True
    if hasattr(tf.keras.utils, "register_keras_serializable"):
        cls = tf.keras.utils.register_keras_serializable()(cls)
    return cls
