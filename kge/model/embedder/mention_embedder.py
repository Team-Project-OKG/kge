import os
import pickle
import math

from torch import Tensor
import torch
#import transformers
import numpy as np

from gensim.models import KeyedVectors

from kge import Config, Dataset
from kge.misc import kge_base_dir
from kge.model import LookupEmbedder


class MentionEmbedder(LookupEmbedder):
    """Base class for embedders of the open link prediction task with a fixed number of objects.

    Objects are mentions of entities and relations, each associated with a sequence of tokens.

    Consists of two embedding layers. To obtain the base embeddings of tokens, the functionality of LookupEmbedder is utilized.
    The MentionEmbedder itself specifies how a sequence of token embeddings is combined to obtain a mention embedding.

    """

    # save pretrained embedding model in class attribute to only load it once
    _pretrained_model = None
    _pretrained_model_file = None

    _n_precached_embeddings = None
    _precached_embeddings = None

    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key: str,
            vocab_size: int,
            init_for_load_only=False,
    ):
        super().__init__(config, dataset, configuration_key, vocab_size, init_for_load_only=init_for_load_only)

        enable_bpe = config.get("dataset.byte_pair_encoding")
        self._bin_batch = self.get_option("bin_within_batch")
        self._bin_size = self.get_option("bin_size")

        if "relation" in self.configuration_key:
            # return token or sub-token sequence
            self._token_lookup = self.dataset.get_mention_to_token_id_map("relation").to(self.config.get("job.device"))
        elif "entity" in self.configuration_key:
            self._token_lookup = self.dataset.get_mention_to_token_id_map("entity").to(self.config.get("job.device"))
        else:
            raise NameError(f"Key '{self.configuration_key}' has to contain 'entity or 'relation'!")
        self._cut_padding = self.get_option("cut_padding_in_batch")

        if self.get_option("pretrained.use"):
            self._init_pretrained_word_emb()

        self._padding_indexes = self.config.get("dataset.padding_indexes")
        self._reset_padding = self.get_option("set_padding_embeddings_to_0")
        self.reset_padding_index()

        if self.get_option("token_embedding_model.use"):
            del self._embeddings
            self._init_token_embedding_model()

        if self.get_option("token_embedding_model.precache"):
            self._n_precached_embeddings = self.get_option("token_embedding_model.precache")
            self._init_precache()


    # set embeddings weights at padding, mention start and mention end index to 0
    def reset_padding_index(self):
        if self._reset_padding:
            self._embeddings.weight.data[self._padding_indexes] = 0

    def lookup_tokens(self, indexes: Tensor) -> Tensor:
        token_seq = self._token_lookup[indexes]
        if self._cut_padding:
            return token_seq[:, 0:torch.max((token_seq > 0).sum(dim=1)).item()]
        else:
            return token_seq

    def embed_tokens(self, token_indexes: Tensor) -> Tensor:
        if self.get_option("token_embedding_model.use"):
            if self._precached_embeddings is not None:
                original_token_indexes = token_indexes.clone()
                original_shape = token_indexes.shape
                replacement_index = token_indexes[:,0] < 0
                precached_indexes = token_indexes[replacement_index][:,0]* -1 - 1
                token_indexes = token_indexes[replacement_index == False]
                new_embeddings = torch.empty([original_shape[0], original_shape[1], self.dim], device=token_indexes.device)

            with torch.no_grad():
                embeddings = self._pretrained_model(token_indexes, (~ (token_indexes == 0)))[0] * ((~ (token_indexes == 0))[..., None])

                if self._precached_embeddings is not None:
                    new_embeddings[replacement_index == False] = embeddings
                    lookup = self._precached_embeddings[precached_indexes][:, :original_shape[1]]
                    new_embeddings[replacement_index] = lookup

                    embeddings = new_embeddings

                return embeddings
        else:
            return self._embeddings(token_indexes.long())

    def embed(self, indexes: Tensor) -> Tensor:
        if self._bin_batch:
            token_indexes = self.lookup_tokens(indexes)
            seq_lengths = (~(token_indexes == 0)).sum(dim=1).cpu().data.numpy()
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

    def _init_pretrained_word_emb(self):
        use_pickle = self.get_option("pretrained.use_pickle")
        filename = self.get_option("pretrained.file.name")
        filetype = self.get_option("pretrained.file.type")
        oov_tactic = self.get_option("pretrained.oov_tactic")
        if "relation" in self.configuration_key:
            part_type = "relation"
        elif "entity" in self.configuration_key:
            part_type = "entity"
        folder = os.path.join(kge_base_dir(), "pretrained")
        # load pickle specific to current dataset and selected embeddings
        if use_pickle:
            pickle_filename = os.path.join(
                folder,
                filename + f"_{self.config.get('dataset.name')}_{part_type}_{oov_tactic}.pckl"
            )
            if os.path.isfile(pickle_filename):
                with open(pickle_filename, "rb") as f:
                    self._embeddings = pickle.load(f)
                self._embeddings.weight.requires_grad = not self.get_option("pretrained.freeze")
                self.dim = self._embeddings.embedding_dim
                self.config.log(f"Loaded pretrained embeddings from {pickle_filename}")
                return
        # create embeddings tensor from scratch
        if MentionEmbedder._pretrained_model is None or MentionEmbedder._pretrained_model_file != filename:
            MentionEmbedder._pretrained_model = KeyedVectors.load_word2vec_format(
                os.path.join(folder, filename + "." + filetype),
                binary=(filetype == "bin")
            )
            MentionEmbedder._pretrained_model_file = filename
        token_list = self.dataset._meta[f'{part_type}_token_ids']
        oov_random = oov_tactic == "random"
        oov_counter = 0
        if oov_random and self._embeddings.embedding_dim != MentionEmbedder._pretrained_model.vector_size:
            self.dim = MentionEmbedder._pretrained_model.vector_size
            # reinit embeddings with correct size
            self._embeddings = torch.nn.Embedding(
                self.vocab_size, self.dim, sparse=self.sparse,
            )
            self.config.log(f"Readjusted embedding size to {self.dim} according to pretrained embeddings {filename}")
        for (i, token) in zip(range(len(token_list)), token_list):
            try:
                self._embeddings.weight.data[i] = torch.from_numpy(MentionEmbedder._pretrained_model.get_vector(token).copy())
            except KeyError:
                try:
                    self._embeddings.weight.data[i] = torch.from_numpy(
                        MentionEmbedder._pretrained_model.get_vector(token.capitalize()).copy())
                except (KeyError, AttributeError):  # attribute error if token is None
                    oov_counter += 1
                    if oov_random:
                        continue
                    # TODO: integrate byte-pair encoding
        self._embeddings.weight.requires_grad = not self.get_option("pretrained.freeze")
        self.config.log(f"Initialized embeddings based on {filename}. OOV Errors: {oov_counter} (Rate: "
                        f"{(oov_counter / len(token_list) * 100):.2f}%)")
        if use_pickle:
            Dataset._pickle_dump_atomic(self._embeddings, pickle_filename)

    def _init_token_embedding_model(self):
        if MentionEmbedder._pretrained_model is None:
            MentionEmbedder._pretrained_model = transformers.AutoModel.from_pretrained(
                self.get_option("token_embedding_model.name")).to(self.config.get("job.device"))

    def _init_precache(self):
        batch_size = self.config.get("train.batch_size")
        embeddings_list = []
        for batch_number in range(math.ceil(self._n_precached_embeddings / batch_size)):
            token_indexes = self._token_lookup[batch_number * batch_size:min((batch_number + 1) * batch_size, self._n_precached_embeddings)]
            embeddings = self.embed_tokens(token_indexes)
            embeddings_list.append(embeddings)

        self._token_lookup[0:self._n_precached_embeddings, 0] = torch.arange(start=-1, end=-self._n_precached_embeddings-1, step=-1)

        self._precached_embeddings = torch.cat(embeddings_list).to(self.config.get("job.device"))
        del embeddings_list
        del embeddings
        del token_indexes
