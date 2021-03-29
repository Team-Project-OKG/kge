from torch import Tensor
from kge import Config, Dataset
from kge.model import MentionEmbedder
import torch

class BigramLookupEmbedder(MentionEmbedder):
    """Bigram mention embedder with max, mean, sum pooling."""

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
        self.pooling = self.get_option("pooling")

        if "relation" in self.configuration_key:
            self.encoder = torch.nn.Conv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=2, dilation=1, bias=False)
        if "entity" in self.configuration_key:
            self.encoder = torch.nn.Conv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=2, dilation=1, bias=False)

    def _token_embed(self, token_indexes: Tensor):
        token_mask = (~(token_indexes == 0)).unsqueeze(1).float()[:, :, 1:]
        token_embeddings = self.embed_tokens(token_indexes).transpose(1, 2)
        encoded = self.encoder(token_embeddings)
        encoded = encoded + token_embeddings[:, :, 1:]
        # pooling on token embeddings
        if self.pooling == 'max':
            encoded = encoded.transpose(1, 2)
            encoded[token_mask.squeeze(1) == 0] = -float("inf")
            pooled = encoded.max(dim=1).values
        elif self.pooling == 'sum':
            pooled = (encoded * token_mask).sum(dim=2)
        elif self.pooling == 'mean':
            pooled = (encoded * token_mask).sum(dim=2) / token_mask.sum(dim=2)
        else:
            raise NotImplementedError
        return pooled