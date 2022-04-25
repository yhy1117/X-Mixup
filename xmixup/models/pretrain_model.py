"""Pretrained model"""
import os
import re
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import h5py
from tensorflow.python.keras.saving import hdf5_format
from xmixup.configuration.auto import PretrainedConfig
from xmixup.configuration.xlmr import RobertaConfig
from xmixup.utils.model_utils import get_initializer
from xmixup.utils.file_utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, is_remote_url, cached_path, is_offline_mode, \
    hf_bucket_url
from xmixup.utils.tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model

logger = logging.getLogger(__name__)


def load_tf_weights(model, resolved_archive_file, _prefix=None):
    """
    Detect missing and unexpected layers and load the TF weights accordingly to their names and shapes.

    Args:
        model (:obj:`tf.keras.models.Model`):
            The model to load the weights into.
        resolved_archive_file (:obj:`str`):
            The location of the H5 file.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    missing_layers = []
    unexpected_layers = []

    # Read the H5 file
    with h5py.File(resolved_archive_file, "r") as f:
        # Retrieve the name of each layer from the H5 file
        saved_h5_model_layers_name = set(hdf5_format.load_attributes_from_hdf5_group(f, "layer_names"))

        # Find the missing layers from the high level list of layers
        missing_layers = list(set([layer.name for layer in model.layers]) - saved_h5_model_layers_name)

        # Find the unexpected layers from the high level list of layers
        unexpected_layers = list(saved_h5_model_layers_name - set([layer.name for layer in model.layers]))
        saved_weight_names_set = set()
        symbolic_weights_names = set()
        weight_value_tuples = []

        # Compute missing and unexpected sub layers
        # Store the weights in list of tuples that looks like [(weight_object, value_of_weight),...]
        for layer in model.layers:
            # if layer_name from the H5 file belongs to the layers from the instantiated model
            if layer.name in saved_h5_model_layers_name:
                # Get the H5 layer object from its name
                h5_layer_object = f[layer.name]
                # Get all the weights as a list from the layer object
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                saved_weights = {}

                # Create a dict from the H5 saved model that looks like {"weight_name": weight_value}
                # And a set with only the names
                for weight_name in hdf5_format.load_attributes_from_hdf5_group(h5_layer_object, "weight_names"):
                    # TF names always start with the model name so we ignore it
                    name = "/".join(weight_name.split("/")[1:])

                    if _prefix is not None:
                        name = _prefix + "/" + name

                    saved_weights[name] = np.asarray(h5_layer_object[weight_name])

                    # Add the updated name to the final list for computing missing/unexpected values
                    saved_weight_names_set.add(name)

                # Loop over each weights from the instantiated model and compare with the weights from the H5 file
                for symbolic_weight in symbolic_weights:
                    # TF names always start with the model name so we ignore it
                    if _prefix is not None:
                        delimeter = len(_prefix.split("/"))
                        symbolic_weight_name = "/".join(
                            symbolic_weight.name.split("/")[:delimeter]
                            + symbolic_weight.name.split("/")[delimeter + 1 :]
                        )
                    else:
                        symbolic_weight_name = "/".join(symbolic_weight.name.split("/")[1:])

                    # here we check if the current weight is among the weights from the H5 file
                    # If yes, get the weight_value of the corresponding weight from the H5 file
                    # If not, make the value to None
                    saved_weight_value = saved_weights.get(symbolic_weight_name, None)

                    # Add the updated name to the final list for computing missing/unexpected values
                    symbolic_weights_names.add(symbolic_weight_name)

                    # If the current weight is found
                    if saved_weight_value is not None:
                        # Check if the shape of the current weight and the one from the H5 file are different
                        if K.int_shape(symbolic_weight) != saved_weight_value.shape:
                            # If yes we reshape the weight from the H5 file accordingly to the current weight
                            # If the two shapes are not compatible we raise an issue
                            try:
                                array = np.reshape(saved_weight_value, K.int_shape(symbolic_weight))
                            except AssertionError as e:
                                e.args += (K.int_shape(symbolic_weight), saved_weight_value.shape)
                                raise e
                        else:
                            array = saved_weight_value

                        # We create the tuple that will be loaded and add it to the final list
                        weight_value_tuples.append((symbolic_weight, array))

    # Load all the weights
    K.batch_set_value(weight_value_tuples)

    # Compute the missing and unexpected layers
    missing_layers.extend(list(symbolic_weights_names - saved_weight_names_set))
    unexpected_layers.extend(list(saved_weight_names_set - symbolic_weights_names))

    return missing_layers, unexpected_layers


class PreTrainedModel(tf.keras.Model):
    r""" Base class for all TF models.

        :class:`~transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods for loading/downloading/saving models
        as well as a few methods common to all models to (i) resize the input embeddings and (ii) prune heads in the self-attention heads.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
            - ``load_tf_weights``: a python ``method`` for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:

                - ``model``: an instance of the relevant subclass of :class:`~transformers.PreTrainedModel`,
                - ``config``: an instance of the relevant subclass of :class:`~transformers.PretrainedConfig`,
                - ``path``: a path (string) to the TensorFlow checkpoint.

            - ``base_model_prefix``: a string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = None
    base_model_prefix = ""

    @property
    def dummy_inputs(self):
        # Sometimes XLM has language embeddings so don't forget to build them as well if needed
        inputs_list = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        if hasattr(self.config, "n_langs") and self.config.n_langs > 1:
            langs_list = tf.constant([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config

    def get_input_embeddings(self):
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`tf.keras.layers.Layer`:
                A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def set_input_embeddings(self, value):
        """
        Set model's input embeddings

        Args:
            value (:obj:`tf.keras.layers.Layer`):
                A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        """
        Returns the model's output embeddings.

        Returns:
            :obj:`tf.keras.layers.Layer`:
                A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens ``tf.Variable`` Module of the model.

        Return: ``tf.Variable``
            Pointer to the input tokens Embeddings Module of the model
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens):
        # get_input_embeddings and set_input_embeddings need to be implemented in base layer.
        base_model = getattr(self, self.base_model_prefix, self)
        old_embeddings = base_model.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        base_model.set_input_embeddings(new_embeddings)
        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens
        return base_model.get_input_embeddings()

    def _get_word_embeddings(self, embeddings):
        if hasattr(embeddings, "word_embeddings"):
            # TFBertEmbeddings, TFAlbertEmbeddings, TFElectraEmbeddings
            return embeddings.word_embeddings
        elif hasattr(embeddings, "weight"):
            # TFSharedEmbeddings
            return embeddings.weight
        else:
            raise ValueError("word embedding is not defined.")

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Variable from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end.

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``tf.Variable``
            Pointer to the resized word Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        word_embeddings = self._get_word_embeddings(old_embeddings)
        if new_num_tokens is None:
            return word_embeddings
        old_num_tokens, old_embedding_dim = word_embeddings.shape
        if old_num_tokens == new_num_tokens:
            return word_embeddings

        # initialize new embeddings
        # todo: initializer range is not always passed in config.
        init_range = getattr(self.config, "initializer_range", 0.02)
        new_embeddings = self.add_weight(
            "weight",
            shape=[new_num_tokens, old_embedding_dim],
            initializer=get_initializer(init_range),
            dtype=tf.float32,
        )
        init_weights = new_embeddings.numpy()

        # Copy token embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        init_weights[:num_tokens_to_copy] = word_embeddings[:num_tokens_to_copy, :]
        new_embeddings.assign(init_weights)

        return new_embeddings

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and associated values being the list of heads to prune in said layer (list of `int`).
        """
        raise NotImplementedError

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the :func:`~transformers.PreTrainedModel.from_pretrained` class method.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save configuration file
        self.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)
        self.save_weights(output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (:obj:`str`, `optional`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `PyTorch state_dict save file` (e.g, ``./pt_model/pytorch_model.bin``). In
                      this case, ``from_pt`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the PyTorch model in a
                      TensorFlow model using the provided conversion scripts and loading the TensorFlow model
                      afterwards.
                    - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments ``config`` and ``state_dict``).
            model_args (sequence of positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            config (:obj:`Union[PretrainedConfig, str]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            from_pt: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a PyTorch state_dict save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies: (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try doanloading the model).
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            mirror(:obj:`str`, `optional`):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
                automatically loaded:

                    - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                      underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                      initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                      ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's ``__init__`` function.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.

        Examples::

            >>> from transformers import BertConfig, TFBertModel
            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = TFBertModel.from_pretrained('bert-base-uncased')
            >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
            >>> model = TFBertModel.from_pretrained('./test/saved_model/')
            >>> # Update configuration during loading.
            >>> model = TFBertModel.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> assert model.config.output_attentions == True
            >>> # Loading from a Pytorch model file instead of a TensorFlow checkpoint (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file('./pt_model/my_pt_model_config.json')
            >>> model = TFBertModel.from_pretrained('./pt_model/my_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        load_weight_prefix = kwargs.pop("load_weight_prefix", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "model", "framework": "tensorflow", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint in priority if from_pt
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        f"Error no file named {[WEIGHTS_NAME, TF2_WEIGHTS_NAME]} found in directory "
                        f"{pretrained_model_name_or_path} or `from_pt` set to False"
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(WEIGHTS_NAME if from_pt else TF2_WEIGHTS_NAME),
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {TF2_WEIGHTS_NAME}, {WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)
            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if from_pt:

            # Load from a PyTorch checkpoint
            return load_pytorch_checkpoint_in_tf2_model(model, resolved_archive_file, allow_missing_keys=True)

        # we might need to extend the variable scope for composite models
        if load_weight_prefix is not None:
            with tf.compat.v1.variable_scope(load_weight_prefix):
                model(model.dummy_inputs, training=False)  # build the network with dummy inputs
        else:
            model(model.dummy_inputs, training=False)  # build the network with dummy inputs

        assert os.path.isfile(resolved_archive_file), f"Error retrieving file {resolved_archive_file}"
        # 'by_name' allow us to do transfer learning by skipping/adding layers
        # see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1339-L1357
        try:
            missing_keys, unexpected_keys = load_tf_weights(model, resolved_archive_file, load_weight_prefix)
        except OSError:
            raise OSError(
                "Unable to load weights from h5 file. "
                "If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. "
            )

        model(model.dummy_inputs)  # Make sure restore ops are run

        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some layers from the model checkpoint at {pretrained_model_name_or_path} were not used when "
                f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n"
                f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.warning(f"All model checkpoint layers were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some layers of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                f"and are newly initialized: {missing_keys}\n"
                f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        else:
            logger.warning(
                f"All the layers of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                f"If your task is similar to the task the model of the checkpoint was trained on, "
                f"you can already use {model.__class__.__name__} for predictions without further training."
            )

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

            return model, loading_info

        return model


class RobertaPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
