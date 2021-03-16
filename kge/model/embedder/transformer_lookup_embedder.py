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
        self._dropout = 0.1
        self._dimensions = self.get_option("dim")
        self._nheads = self.get_option("nhead")
        self._dim_ff_layer = self.get_option("dim_ff")
        self._num_layers = self.get_option("num_layers")
        self._layer_norm = nn.LayerNorm(self._dimensions)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self._dimensions, nhead=self._nheads,
                                                         dim_feedforward=self._dim_ff_layer, dropout=self._dropout)
        self._encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, self._num_layers, self._layer_norm)
        self.pos_encoder = PositionalEncoding(self._dimensions, 100, 0.1)
        self._decoder = nn.Linear(self._dimensions, self._dimensions)
        self._device = self.config.get("job.device")

        for layer in self._encoder_transformer.layers:
            self.initialize(layer.linear1.weight.data)
            self.initialize(layer.linear2.weight.data)
            self.initialize(layer.self_attn.out_proj.weight.data)

            if layer.self_attn._qkv_same_embed_dim:
                self.initialize(layer.self_attn.in_proj_weight)
            else:
                self.initialize(layer.self_attn.q_proj_weight)
                self.initialize(layer.self_attn.k_proj_weight)
                self.initialize(layer.self_attn.v_proj_weight)
                
    def _token_embed(self, token_indexes):

        token_embeddings = self.embed_tokens(token_indexes.long()).permute(1,0,2)
        src_key_padding_mask = (token_indexes == 0)
        transformer_input = token_embeddings * math.sqrt(self._dimensions)
        transformer_input = self.pos_encoder(transformer_input)
        transformer_input = (~src_key_padding_mask * transformer_input.permute(2, 1, 0)).permute(2,1,0)
        # src_mask = torch.nn.Transformer.generate_square_subsequent_mask(self, transformer_input.size()[0]).to(
        #    self._device)
        encoded = self._encoder_transformer(transformer_input,
                                            src_key_padding_mask=src_key_padding_mask)
        encoded = (~src_key_padding_mask * encoded.permute(2, 1, 0)).permute(2,1,0)
        decoded = self._decoder(encoded)
        decoded = (~src_key_padding_mask * decoded.permute(2, 1, 0)).permute(1,2,0)

        if self._pooling == 'max':  # should reduce dimensions to (batch_size, dim)
            pooled_embeddings = decoded.max(dim=1).values
        elif self._pooling == 'mean':
            lengths = (token_indexes > 0).sum(dim=1)
            pooled_embeddings = decoded.sum(dim=1) / lengths.unsqueeze(1)
        elif self._pooling == 'sum':
            pooled_embeddings = decoded.sum(dim=1)
        elif self._pooling == 'pos':
            pooled_embeddings = decoded[:,0,:]
        else:
            raise NotImplementedError
        return pooled_embeddings


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, dropout=0.1):
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


''' def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key: str,
            vocab_size: int,
            init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, vocab_size, init_for_load_only=init_for_load_only)

        self._dimensions = self.get_option("dim")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self._dimensions, nhead=self.get_option("nhead"), dim_feedforward = self.get_option("dim_ff"))
        self._encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, self.get_option("num_layers"))

    def _token_embed(self, token_indexes):
        #switch batch and sequence dimension to match input format
        token_embeddings = self.embed_tokens(token_indexes.long())
        transformer_input = token_embeddings.permute(1, 0, 2)
        last_state = (token_indexes > 0).sum(dim = 1) - 1
        src_padding_mask = (token_indexes == 0)
        encoded = self._encoder_transformer(transformer_input, src_key_padding_mask = src_padding_mask).permute(1, 0, 2)
        return encoded[torch.arange(0, encoded.shape[0]), last_state]

    #def embed(self, indexes: Tensor) -> Tensor:
    #    return self._forward(super().embed(indexes), self._token_lookup[indexes])

    # return the pooled token entity/relation embedding vectors
   # def embed_all(self) -> Tensor:
   #     return self._forward(super().embed_all(), self._token_lookup) '''
