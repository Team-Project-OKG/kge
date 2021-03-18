from torch import Tensor
import torch.nn

from kge import Config, Dataset
from kge.model import MentionEmbedder


class PaddingLstmLookupEmbedder(MentionEmbedder):

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
            dropout=0,
            batch_first=True,
            num_layers=self.num_layers
        )

    def _token_embed(self, token_indexes):
        token_embeddings = self.embed_tokens(token_indexes.long())
        lstm_output, hn = self._encoder_lstm(token_embeddings)
        num_tokens = (~(token_indexes == 0)).sum(dim=1)
        return lstm_output[:, -1, :]
