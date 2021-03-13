from torch import Tensor
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from kge import Config, Dataset
from kge.model import MentionEmbedder


class PackedLstmLookupEmbedder(MentionEmbedder):

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

        if self.get_option("emb_dim_as_hidden_dim"):
            self.hidden_dim = self.dim
        else:
            self.hidden_dim = self.get_option("hidden_dim")

        self.num_layers = self.get_option("num_layers")

        self._encoder_lstm = torch.nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            dropout=0,
            num_layers=self.num_layers
        )

    def _token_embed(self, token_indexes):
        token_embeddings = self.embed_tokens(token_indexes.long())
        lengths = (~ (token_indexes == 0)).sum(dim=1).cpu()
        padded_sequence = pack_padded_sequence(token_embeddings, lengths, batch_first=True, enforce_sorted=False)
        padded_output, _ = self._encoder_lstm(padded_sequence)
        output, _ = pad_packed_sequence(padded_output, batch_first=True)
        return output[torch.arange(0, output.shape[0]), lengths - 1]
