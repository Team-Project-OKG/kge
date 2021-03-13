import os
import pickle
import math

from torch import Tensor
import torch
import transformers
import numpy as np

from gensim.models import KeyedVectors

from kge import Config, Dataset
from kge.misc import kge_base_dir
from kge.model import LookupEmbedder
from kge.util.byte_pair_encoding import BPESubTokenEmbedder


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
            self._token_lookup = self.dataset._mentions_to_token_ids["relations"].to(self.config.get("job.device"))
        elif "entity" in self.configuration_key:
            self._token_lookup = self.dataset._mentions_to_token_ids["entities"].to(self.config.get("job.device"))

        if enable_bpe:
            self._sub_token_embedder = BPESubTokenEmbedder(dataset.bpe_vocab, configuration_key)

        if self.get_option("pretrained.use"):
            self._init_pretrained_word_emb()
        self._padding_indexes = self.config.get("dataset.padding_indexes")
        self.reset_padding_index()

        if self.get_option("token_embedding_model.use"):
            del self._embeddings
            self._init_token_embedding_model()

        if self.get_option("token_embedding_model.precache"):
            self._n_precached_embeddings = self.get_option("token_embedding_model.precache")
            self._init_precache()


    # set embeddings weights at padding, mention start and mention end index to 0
    def reset_padding_index(self):
        self._embeddings.weight.data[self._padding_indexes] = 0

    def lookup_tokens(self, indexes: Tensor) -> Tensor:
        token_seq = self._token_lookup[indexes]
        return token_seq[:, 0:torch.max(torch.nonzero(token_seq), dim=0).values[1]+1]

    def embed_tokens(self, token_indexes: Tensor) -> Tensor:
        # Additionally split up tokens into sub-tokens and embed them
        if self.config.get("dataset.byte_pair_encoding"):
            sub_token_indexes = self._sub_token_embedder.get_sub_tokens_from_tokens(token_indexes)
            emb = self._embeddings(sub_token_indexes)
            return emb
        elif self.get_option("token_embedding_model.use"):
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
                self._embeddings.weight.data[i] = torch.from_numpy(MentionEmbedder._pretrained_model.get_vector(token))
            except KeyError:
                try:
                    self._embeddings.weight.data[i] = torch.from_numpy(
                        MentionEmbedder._pretrained_model.get_vector(f"ENTITY/{token}"))
                except KeyError:
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
        if "distilbert" in self.get_option("token_embedding_model.name"):
            if MentionEmbedder._pretrained_model is None:
                model_class, pretrained_weights = (transformers.DistilBertModel, self.get_option("token_embedding_model.name"))
                MentionEmbedder._pretrained_model = model_class.from_pretrained(pretrained_weights).to(self.config.get("job.device"))

        elif "albert" in self.get_option("token_embedding_model.name"):
            if MentionEmbedder._pretrained_model is None:
                model_class, pretrained_weights = (transformers.AlbertModel, self.get_option("token_embedding_model.name"))
                MentionEmbedder._pretrained_model = model_class.from_pretrained(pretrained_weights).to(self.config.get("job.device"))
        elif "bert" in self.get_option("token_embedding_model.name"):
            if MentionEmbedder._pretrained_model is None:
                model_class, pretrained_weights = (transformers.BertModel, self.get_option("token_embedding_model.name"))
                MentionEmbedder._pretrained_model = model_class.from_pretrained(pretrained_weights).to(self.config.get("job.device"))


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
