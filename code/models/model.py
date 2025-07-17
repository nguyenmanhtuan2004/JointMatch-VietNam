import torch
import torch.nn as nn
from torch.nn.functional import normalize
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLayer, BertEmbeddings, BertPooler
from transformers import AutoModel


class TextClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(TextClassifier, self).__init__()
        # Thay đổi từ BertModel sang PhoBERT
        self.bert = AutoModel.from_pretrained('vinai/phobert-base')
        self.linear = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, num_labels)
        )
        
    def forward(self, inputs):
        outputs = self.bert(**inputs)
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
        # Lấy mean pooling trên chiều seq_len
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        predict = self.linear(pooled_output)
        return predict
