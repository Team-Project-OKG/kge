from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.model import KgeEmbedder

class UnigramPoolingEmbedder(KgeEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only)

        self.dim = self.get_option("dim")
        self.sparse = self.get_option("sparse")
        self.pooling = self.get_option("pooling")
        # TODO: Add entity and relation dropout later
        #self.entity_dropout = entity_dropout if entity_dropout else dropout
        #self.relation_dropout = relation_dropout if relation_dropout else dropout

        if "relation" in self.configuration_key: # Todo: remove + 3 for tokens [Unseen]: 0, [begin]: 1 and [end]: 2
            self._embeddings = torch.nn.Embedding(self.dataset.num_tokens_relations() + 3, self.dim, sparse=self.sparse)
        elif "entity" in self.configuration_key:
            self._embeddings = torch.nn.Embedding(self.dataset.num_tokens_entities() + 3, self.dim, sparse=self.sparse)


        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0, "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)

    def _embed(self, token_indexes: Tensor) -> Tensor:
        return self._embeddings(token_indexes)

    def _embeddings_all(self):
        return self._embeddings

    def embed(self, indexes: Tensor) -> Tensor:
        if "relation" in self.configuration_key:
            token_indexes = self.dataset.relation_mentions_to_token_ids()[indexes].to(self.config.get("job.device"))
        elif "entity" in self.configuration_key:
            token_indexes = self.dataset.entity_mentions_to_token_ids()[indexes].to(self.config.get("job.device"))
        else:
            raise NotImplementedError
        # Todo: check if 1 and 2 are omitted beforehand (in OLPDataset)
        #token_indexes = self.map_to_tokens(indexes) #[:,1:-1] # How to handle beginning and end for a relation([1,...2])? Omit for now

        # lookup all tokens -> token embeddings with expected shape: 3D tensor (batch_size, max_tokens, dim)
        token_embeddings = self._embed(token_indexes)
        # pooling on token embeddings
        if self.pooling == 'max':  # should reduce dimensions to (batch_size, dim)
            pooled_embeddings = token_embeddings.max(dim=1).values
        elif self.pooling == 'mean':
            lengths = (token_indexes > 0).sum(dim=1)
            pooled_embeddings = token_embeddings.sum(dim=1) / lengths.unsqueeze(1)
        elif self.pooling == 'sum':
            pooled_embeddings = token_embeddings.sum(dim=1)
        else:
            raise NotImplementedError

        #if self.normalize == 'norm':
        #    token_embeddings = torch.nn.functional.normalize(token_embeddings, dim=1) # l_p normalization
        #if self.normalize == 'batchnorm':
        #if dropout > 0:
        return pooled_embeddings

    # return the pooled token entity/relation embedding vectors
    def embed_all(self) -> Tensor:
        # Todo: remove + 3 for tokens [Unseen]: 0, [begin]: 1 and [end]: 2
        if "relation" in self.configuration_key:
            token_indexes = self.dataset._mentions_to_token_ids["relations"].to(self.config.get("job.device"))

        elif "entity" in self.configuration_key:
            token_indexes = self.dataset._mentions_to_token_ids["entities"].to(self.config.get("job.device"))
        else:
            raise NotImplementedError

        token_embeddings = self._embed(token_indexes)
        if self.pooling == 'max':  # should reduce dimensions to (batch_size, dim)
            pooled_embeddings = token_embeddings.max(dim=1).values
        elif self.pooling == 'mean':
            lengths = (token_indexes > 0).sum(dim=1)
            pooled_embeddings = token_embeddings.sum(dim=1) / lengths.unsqueeze(1)
        elif self.pooling == 'sum':
            pooled_embeddings = token_embeddings.sum(dim=1)
        else:
            raise NotImplementedError
        return pooled_embeddings  # Todo: put to device with: .cuda()
