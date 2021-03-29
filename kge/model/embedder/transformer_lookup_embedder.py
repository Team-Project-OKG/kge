from torch import Tensor
import torch.nn as nn
import torch
import math
from kge import Config, Dataset
from kge.model import MentionEmbedder


class TransformerLookupEmbedder(MentionEmbedder):

    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key: str,
            vocab_size: int,
            init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, vocab_size, init_for_load_only=init_for_load_only)
        self._pooling = self.get_option("pooling")
        if self._pooling == "cls":
            self.cls_emb = torch.nn.parameter.Parameter(torch.zeros(self.dim))

        self._transformer_dropout = self.get_option("transformer_dropout")
        self._nheads = self.get_option("nhead")
        self._dim_ff_layer = self.get_option("dim_ff")
        self._num_layers = self.get_option("num_layers")
        self._layer_norm = nn.LayerNorm(self.dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.dim, nhead=self._nheads,
                                                         dim_feedforward=self._dim_ff_layer, dropout=self._transformer_dropout)
        self._encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, self._num_layers, self._layer_norm)

        self._maxlen = self._token_lookup.shape[1]
        if self._pooling == "cls":
            self._maxlen += 1
        self.pos_encoder = PositionalEncoding(self.dim, self._maxlen, self._transformer_dropout)

    def _token_embed(self, token_indexes):
        # prepare input for transformer encoder
        token_embeddings = self.embed_tokens(token_indexes.long()).permute(1, 0, 2)
        src_key_padding_mask = (token_indexes == 0)

        if self._pooling == 'cls':
            # add cls token at the beginning and adjust src_key_padding_mask
            token_embeddings = torch.cat([self.cls_emb.repeat(1, token_embeddings.shape[1], 1), token_embeddings], dim=0)
            src_key_padding_mask = torch.cat([
                torch.zeros([src_key_padding_mask.shape[0], 1], dtype=torch.bool, device=token_embeddings.device), src_key_padding_mask], dim=1)
        transformer_input = token_embeddings * math.sqrt(self.dim)
        transformer_input = self.pos_encoder(transformer_input)

        # transformer encoder
        encoded = self._encoder_transformer(
            transformer_input, src_key_padding_mask=src_key_padding_mask)
        if self._pooling == 'cls':
            return encoded[0, :, :]
        else:
            # set embeddings of padding to 0
            encoded = (~src_key_padding_mask * encoded.permute(2, 1, 0)).permute(1, 2, 0)
            #encoded = encoded.permute(1,0,2)
            if self._pooling == 'max':
                return encoded.max(dim=1).values
            elif self._pooling == 'mean':
                lengths = (~src_key_padding_mask).sum(dim=1)
                return encoded.sum(dim=1) / lengths.unsqueeze(1)
            elif self._pooling == 'sum':
                return encoded.sum(dim=1)
            else:
                raise NotImplementedError


class PositionalEncoding(nn.Module):

    # create positional encoding vector and add it to the word embeddings

    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



