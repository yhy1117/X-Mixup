"""Auto model class"""
from collections import OrderedDict
from xmixup.configuration.auto import PretrainedConfig
from xmixup.configuration.bert import BertConfig
from xmixup.configuration.xlmr import XLMRobertaConfig
from xmixup.models.bert import BertModel
from xmixup.models.xlmr import XLMRobertaModel
from xmixup.tasks.xlmr.sequence_classification import XMixupForSequenceClassification
from xmixup.tasks.xlmr.qa import XMixupForQuestionAnswering
from xmixup.tasks.xlmr.token_classification import XMixupForTokenClassification
from xmixup.tasks.mbert.sequence_classification import XMixupForMBertSequenceClassification
from xmixup.tasks.mbert.token_classification import XMixupForMBertTokenClassification
from xmixup.tasks.mbert.qa import XMixupForMBertQuestionAnswering


MODEL_MAPPING = OrderedDict(
    [
        (XLMRobertaConfig, XLMRobertaModel),
        (BertConfig, BertModel),
    ]
)

MODEL_FOR_MIXUP_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (XLMRobertaConfig, XMixupForSequenceClassification),
        (BertConfig, XMixupForMBertSequenceClassification),
    ]
)

MODEL_FOR_MIXUP_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        (XLMRobertaConfig, XMixupForQuestionAnswering),
        (BertConfig, XMixupForMBertQuestionAnswering),
    ]
)

MODEL_FOR_MIXUP_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (XLMRobertaConfig, XMixupForTokenClassification),
        (BertConfig, XMixupForMBertTokenClassification),
    ]
)


class AutoModelForTask(object):
    r"""
        :class:`~transformers.AutoModelForTask` is a generic model class
        that will be instantiated as one of the model classes of the library
        when created with the `AutoModelForTask.from_pretrained(pretrained_model_name_or_path)`
        class method.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

            - `distilbert`: TFDistilBertForSequenceClassification (DistilBERT model)
            - `roberta`: TFRobertaForSequenceClassification (RoBERTa model)
            - `bert`: TFBertForSequenceClassification (Bert model)
            - `xlnet`: TFXLNetForSequenceClassification (XLNet model)
            - `xlm`: TFXLMForSequenceClassification (XLM model)

        This class cannot be instantiated using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "TFAutoModelForSequenceClassification is designed to be instantiated "
            "using the `TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`TFAutoModelForSequenceClassification.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, task_name, config):
        r""" Instantiates one of the base model classes of the library
        from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights.
            It only affects the model's configuration. Use :func:`~transformers.TFAutoModel.from_pretrained` to load
            the model weights

        Args:
            task_name: whether the task is sequence classification or token classification or qa or multiple choices.
            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                The model class to instantiate is selected based on the configuration class:

                    - isInstance of `distilbert` configuration class: DistilBertModel (DistilBERT model)
                    - isInstance of `roberta` configuration class: RobertaModel (RoBERTa model)
                    - isInstance of `bert` configuration class: BertModel (Bert model)
                    - isInstance of `xlnet` configuration class: XLNetModel (XLNet model)
                    - isInstance of `xlm` configuration class: XLMModel (XLM model)

        Examples::

            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            model = TFAutoModelForSequenceClassification.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        """
        if task_name == "base":
            MODEL_FOR_MAPPING = MODEL_MAPPING
        elif task_name == "sequence classification":
            MODEL_FOR_MAPPING = MODEL_FOR_MIXUP_SEQUENCE_CLASSIFICATION_MAPPING
        elif task_name == "qa":
            MODEL_FOR_MAPPING = MODEL_FOR_MIXUP_QUESTION_ANSWERING_MAPPING
        elif task_name == "token classification":
            MODEL_FOR_MAPPING = MODEL_FOR_MIXUP_TOKEN_CLASSIFICATION_MAPPING
        else:
            raise NotImplementedError("Now only support sequence classification or token classification or qa or multiple choices task")
        for config_class, model_class in MODEL_FOR_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, task_name, pretrained_model_name_or_path, *model_args, **kwargs):
        r""" Instantiates one of the sequence classification model classes of the library
        from a pre-trained model configuration.

        The `from_pretrained()` method takes care of returning the correct model class instance
        based on the `model_type` property of the config object, or when it's missing,
        falling back to using pattern matching on the `pretrained_model_name_or_path` string:

            - `distilbert`: TFDistilBertForSequenceClassification (DistilBERT model)
            - `roberta`: TFRobertaForSequenceClassification (RoBERTa model)
            - `bert`: TFBertForSequenceClassification (Bert model)
            - `xlnet`: TFXLNetForSequenceClassification (XLNet model)
            - `xlm`: TFXLMForSequenceClassification (XLM model)

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with `model.train()`

        Params:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
                - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

            from_pt: (`Optional`) Boolean
                Set to True if the Checkpoint is a PyTorch checkpoint.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) instance of a class derived from :class:`~transformers.TFPretrainedConfig`:
                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.TFPreTrainedModel.save_pretrained` and :func:`~transformers.TFPreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.TFPretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = TFAutoModelForSequenceClassification.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
            model = TFAutoModelForSequenceClassification.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if task_name == "base":
            MODEL_FOR_MAPPING = MODEL_MAPPING
        elif task_name == "sequence classification":
            MODEL_FOR_MAPPING = MODEL_FOR_MIXUP_SEQUENCE_CLASSIFICATION_MAPPING
        elif task_name == "qa":
            MODEL_FOR_MAPPING = MODEL_FOR_MIXUP_QUESTION_ANSWERING_MAPPING
        elif task_name == "token classification":
            MODEL_FOR_MAPPING = MODEL_FOR_MIXUP_TOKEN_CLASSIFICATION_MAPPING
        else:
            raise NotImplementedError("Now only support sequence classification or token classification or qa or multiple choices task")

        for config_class, model_class in MODEL_FOR_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of TFAutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_MAPPING.keys()),
            )
        )
