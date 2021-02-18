from torch import Tensor
import torch

from kge import Config, Dataset
from kge.model import LookupEmbedder
from kge.util.byte_pair_encoding import BPESubTokenEmbedder

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
        super().__init__(config, dataset, configuration_key, vocab_size, init_for_load_only=init_for_load_only)

        enable_bpe = True #config.get("dataset.byte_pair_encoding")
        if "relation" in self.configuration_key:
            self._token_lookup = self.dataset._mentions_to_token_ids["relations"].to(self.config.get("job.device"))
        elif "entity" in self.configuration_key:
            self._token_lookup = self.dataset._mentions_to_token_ids["entities"].to(self.config.get("job.device"))

        if enable_bpe:
            self._sub_token_embedder = BPESubTokenEmbedder(dataset.bpe_vocab, configuration_key)

    def lookup_tokens(self, indexes: Tensor) -> Tensor: #TODO: must be reviewed
        token_seq = self._token_lookup[indexes]
        return token_seq[:, 0:torch.max(torch.nonzero(token_seq), dim=0).values[1]+1]

    def embed_tokens(self, token_indexes: Tensor) -> Tensor:
        # Additionally split up tokens into sub-tokens and embed them
        if True: #self.config.get("dataset.byte_pair_encoding"):
            sub_token_indexes = self._sub_token_embedder.get_sub_tokens_from_tokens(token_indexes)

            emb = self._embeddings(sub_token_indexes)
            return emb
        else:
            return self._embeddings(token_indexes.long())

    def embed(self, indexes: Tensor) -> Tensor:
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
