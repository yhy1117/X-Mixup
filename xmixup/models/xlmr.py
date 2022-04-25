# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
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
""" TF 2.0  XLM-RoBERTa model. """
from xmixup.models.roberta import RobertaModel
from xmixup.configuration.xlmr import XLMRobertaConfig


class XLMRobertaModel(RobertaModel):
    """
    This class overrides :class:`~transformers.TFRobertaModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
