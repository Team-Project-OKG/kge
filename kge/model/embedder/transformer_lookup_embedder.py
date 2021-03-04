from torch import Tensor
import torch.nn

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

        self._dimensions = self.get_option("dim")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self._dimensions, nhead=self.get_option("nhead"), dim_feedforward=512)
        self._encoder_transformer = torch.nn.TransformerEncoder(encoder_layer, self.get_option("num_layers"))

    def _token_embed(self, token_indexes):
        #switch batch and sequence dimension to match input format
        token_embeddings = self.embed_tokens(token_indexes.long())
        transformer_input = token_embeddings.permute(1, 0, 2)
        last_state = (token_indexes > 0).sum(dim= 1) - 1
        #src_padding_mask = (token_indexes == 0)
        encoded = self._encoder_transformer(transformer_input).permute(1, 0, 2)
        return encoded[torch.arange(0, encoded.shape[0]), last_state]


    """def _token_embed(self, token_indexes):
        #switch batch and sequence dimension to match input format
        token_embeddings = self.embed_tokens(token_indexes.long())
        transformer_input = token_embeddings.permute(1, 0, 2)
        return self._encoder_transformer(transformer_input)[token_indexes[0].size()[0] - 1]

    #def embed(self, indexes: Tensor) -> Tensor:
    #    return self._forward(super().embed(indexes), self._token_lookup[indexes])

    # return the pooled token entity/relation embedding vectors
   # def embed_all(self) -> Tensor:
   #     return self._forward(super().embed_all(), self._token_lookup)"""