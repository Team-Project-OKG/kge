from torch import Tensor
import torch
import numpy as np

from kge import Config, Dataset
from kge.model import LookupEmbedder

class MentionEmbedder(LookupEmbedder):
    r"""Base class for embedders of the open link prediction task with a fixed number of objects.

    Objects are mentions of entities and relations, each associated with a sequence of tokens.

    Consists of two embedding layers. To obtain the base embeddings of tokens, the functionality of LookupEmbedder is utilized.
    The MentionEmbedder itself specifies how a sequence of token embeddings is combined to obtain a mention embedding.

    """

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

        self.reset_padding_index()

        self._bin_batch = self.get_option("bin_within_batch")
        self._bin_size = self.get_option("bin_size")

        if "relation" in self.configuration_key:
            self._token_lookup = self.dataset._mentions_to_token_ids["relations"].to(self.config.get("job.device"))
        elif "entity" in self.configuration_key:
            self._token_lookup = self.dataset._mentions_to_token_ids["entities"].to(self.config.get("job.device"))

    # set embeddings weights at padding index to 0
    def reset_padding_index(self):
        self._embeddings.weight.data[0] = 0

    def lookup_tokens(self, indexes: Tensor) -> Tensor: #TODO: must be reviewed
        token_seq = self._token_lookup[indexes]
        return token_seq[:, 0:torch.max(torch.nonzero(token_seq), dim=0).values[1]+1]

    def embed_tokens(self, token_indexes: Tensor) -> Tensor:
        return self._embeddings(token_indexes.long())

    def embed(self, indexes: Tensor) -> Tensor:
        if self._bin_batch:
            token_indexes = self.lookup_tokens(indexes)
            seq_lengths = (token_indexes > 0).sum(dim=1).cpu().data.numpy()
            order = np.argsort(seq_lengths)
            rev_order = np.argsort(order)
            lengths, counts = np.unique(seq_lengths, return_counts=True)
            lower_bound = 0
            bin_size = 0
            bin_embeddings = []
            for (length, count) in zip(lengths, counts):
                bin_size += count
                if bin_size >= self._bin_size or lower_bound + bin_size == indexes.shape[0]:
                    bin_embeddings.append(
                        self._token_embed(token_indexes[order[lower_bound:lower_bound+bin_size]][:, 0:length]))
                    lower_bound += bin_size
                    bin_size = 0
            embeddings = torch.cat(bin_embeddings)[rev_order]
        else:
            token_indexes = self.lookup_tokens(indexes)
            # lookup all tokens -> token embeddings with expected shape: 3D tensor (batch_size, max_tokens, dim)
            embeddings = self._token_embed(token_indexes)
        return self._postprocess(embeddings)

    # return the pooled token entity/relation embedding vectors
    def embed_all(self) -> Tensor:
        embeddings = self._token_embed(self._token_lookup)
        return self._postprocess(embeddings)

    def _token_embed(self, indexes: Tensor):
        "Combine token embeddings to one embedding for a mention."
        raise NotImplementedError
