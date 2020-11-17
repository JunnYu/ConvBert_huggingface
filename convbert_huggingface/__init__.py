# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from transformers.file_utils import is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_convbert import ConvBertConfig
from transformers.tokenization_electra import ElectraTokenizer

if is_tokenizers_available():
    from transformers.tokenization_electra_fast import ElectraTokenizerFast

if is_torch_available():
    from .modeling_convbert import (
        ElectraForMaskedLM as ConvBertForMaskedLM,
        ElectraForMultipleChoice as ConvBertForMultipleChoice,
        ElectraForPreTraining as ConvBertForPreTraining,
        ElectraForQuestionAnswering as ConvBertForQuestionAnswering,
        ElectraForSequenceClassification as ConvBertForSequenceClassification,
        ElectraForTokenClassification as ConvBertForTokenClassification,
        ElectraModel as ConvBertModel,
        ElectraPreTrainedModel as ConvBertPreTrainedModel,
        load_tf_weights_in_convbert,
    )
