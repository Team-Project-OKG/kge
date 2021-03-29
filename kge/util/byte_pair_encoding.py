from kge import Dataset
import torch
from torch import Tensor
import time
import numpy as np
from typing import Tuple


class Bidict(dict):
    """
    Extends native Python dictionary with an inverse lookup by
    keeping track of the inverse dictionary. The original
    standard dict: N-to-1
    bidict[key_1] = value_1, bidict[key_2] = value_2
    inverse dict: 1-to-N
    bidict[value_1] = [key_1, key_2]
    """

    def __init__(self, *args, **kwargs):
        super(Bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(Bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(Bidict, self).__delitem__(key)

    def get_dict(self):
        """ Restores the uni-directional native dictionary."""
        return super(Bidict, self).copy()


class BytePairEncodingVocab:
    """
    Input: list of entity/relation token strings

    Algorithm:
        Split words into characters and append unique end token to each word
        Stop if number of max. iteration or desired number of tokens are reached
            Count frequency of each consecutive byte-pair
            Find most frequent byte-pair
            Merge most frequent byte-pair into new token
            Replace individual character pairs by new, most frequent token in vocab
    Output:
        (ent/rel)_subtoken_lookup: Lookup of token index (int) -> list of subtoken indexes
        num_(ent/rel)_subtokens: unique number of subtokens (int)
        (ent/rel)_subtoken_ids: Vocabulary of subtoken indexes to their string representation (dict)
    """

    def __init__(self,
                 olp_dataset: Dataset,
                 iterations_entities,
                 iterations_relations
                 ):

        self.ent_subtoken_ids = None
        self.rel_subtoken_ids = None
        self.num_ent_subtokens = None
        self.num_rel_subtokens = None
        self.ent_subtoken_lookup = None
        self.rel_subtoken_lookup = None

        # run bpe and init vocab
        self.create_subtoken_vocabs(olp_dataset, iterations_entities, iterations_relations)

    def create_subtoken_vocabs(self, olp_dataset, iterations_entities, iterations_relations):
        entity_tokens_str = olp_dataset._meta['entity_token_ids']
        relation_tokens_str = olp_dataset._meta['relation_token_ids']
        delimiter = -1
        end_subtoken = '</w>'
        device = olp_dataset.config.get("job.device")
        num_special_tokens = 4  # '[unmapped]', '[unseen]', '[begin]', '[end]'
        # exclude these tokens from byte-pair encoding
        special_tokens = {idx: token for idx, token in enumerate(entity_tokens_str[:num_special_tokens])}

        # Byte-Pair-Encoding for entities
        # get vocabulary, add stop word '</w>' at the end of each token
        olp_dataset.config.log("Starting byte-pair encoding for entities...")
        time_ent_start = time.time()
        entity_tokens = [' '.join(x) + ' ' + end_subtoken for x in entity_tokens_str[num_special_tokens:]]
        entity_tokens_1d = np.concatenate([np.array(x.split()) for x in entity_tokens])
        unique_characters = np.unique(entity_tokens_1d)
        # use a bidirectional map to update lookups and their inverse simultaneously
        # map index to sub-tokens and inverse
        index_character_map_ent = Bidict({idx + num_special_tokens: x for idx, x in enumerate(unique_characters)})
        subtokens_ent = torch.Tensor([index_character_map_ent.inverse[x][0] for x in entity_tokens_1d]).to(device)
        self.ent_subtoken_lookup, index_character_map_ent, iter_ent = \
            self.run_bpe(iterations_entities, subtokens_ent, index_character_map_ent, end_subtoken, delimiter,
                         num_special_tokens)
        self.ent_subtoken_lookup = {**{key: [key] for key in special_tokens}, **self.ent_subtoken_lookup}
        self.ent_subtoken_ids = {**special_tokens, **index_character_map_ent.get_dict()}  # add special tokens again
        self.num_ent_subtokens = len(self.ent_subtoken_ids)
        time_ent_end = time.time()
        olp_dataset.config.log(f"Ran {iter_ent} iterations of byte-pair encoding for entities.\n"
                               f"Found {self.num_ent_subtokens} unique subtokens in {time_ent_end - time_ent_start:.2f}s")

        # Byte-Pair-Encoding for relations
        # get vocabulary, add stop word '</w>' at the end of each token
        olp_dataset.config.log(f"Starting byte-pair encoding for relations...")
        time_rel_start = time.time()
        relation_tokens = [' '.join(x) + ' ' + end_subtoken for x in relation_tokens_str[num_special_tokens:]]
        relation_tokens_1d = np.concatenate([np.array(x.split()) for x in relation_tokens])
        rel_unique_chars = np.unique(relation_tokens_1d)
        index_character_map_rel = Bidict({idx + num_special_tokens: x for idx, x in enumerate(rel_unique_chars)})
        subtokens_rel = torch.Tensor([index_character_map_rel.inverse[x][0] for x in relation_tokens_1d]).to(device)
        self.rel_subtoken_lookup, index_character_map_rel, iter_rel = self.run_bpe(iterations_relations, subtokens_rel,
                                                                                   index_character_map_rel,
                                                                                   end_subtoken,
                                                                                   delimiter, num_special_tokens)

        self.rel_subtoken_lookup = {**{key: [key] for key in special_tokens}, **self.rel_subtoken_lookup}
        self.rel_subtoken_ids = {**special_tokens, **index_character_map_rel.get_dict()}
        self.num_rel_subtokens = len(self.rel_subtoken_ids)
        time_rel_end = time.time()
        olp_dataset.config.log(f"Ran {iter_rel} iterations of byte-pair encoding for relations.\n"
                               f"Found {self.num_rel_subtokens} unique subtokens in {time_rel_end - time_rel_start:.2f}s")

    def run_bpe(self, iterations,
                subtokens,
                index_character_map,
                end_subtoken,
                delimiter,
                num_special_tokens,
                verbose=False) -> Tuple[dict, dict, int]:

        iter = 0
        for x in range(iterations):
            iter = x + 1
            # Find most frequent bigram
            replace_bigram = self.get_bigrams(subtokens, index_character_map, delimiter, end_subtoken)
            if replace_bigram == None:
                break
            # Merge most frequent bigram
            subtokens, index_character_map = self.merge_bigrams(subtokens, replace_bigram, index_character_map,
                                                                end_subtoken, num_special_tokens)
            if verbose:
                print("Iteration {} -> merge bigram: ({} {})".format(x,
                                                                     index_character_map[int(replace_bigram[0].item())],
                                                                     index_character_map[
                                                                         int(replace_bigram[1].item())]))
        # keep only unique sub-tokens occuring in sub-token sequences after BPE
        unique_subtokens = torch.unique(subtokens[subtokens >= 0])
        index_character_map = Bidict({int(x.item()): index_character_map[int(x.item())] for x in unique_subtokens})
        # avoid gaps in mapping -> remap indexes to consecutive items
        remap = {key: idx + num_special_tokens for idx, key in enumerate(index_character_map.keys())}
        if set(remap.keys()) != set(remap.values()):
            index_character_map = Bidict({remap[key]: val for key, val in index_character_map.items()})
            # also remap subtokens
            subtokens = torch.Tensor(
                [remap[int(x.item())] if int(x.item()) in remap else int(x.item()) for x in subtokens]).to(
                subtokens.device)
        if end_subtoken in index_character_map.inverse:  # split at end_subtoken idx and delimiter
            end_subtoken_idx = index_character_map.inverse[end_subtoken][0]
            end_idxs = torch.where(torch.logical_or(subtokens == delimiter, subtokens == end_subtoken_idx))[0] + 1
        else:  # split only at delimiter
            end_idxs = torch.where(subtokens == delimiter)[0] + 1
        split_idxs = np.diff(([0] + end_idxs.tolist() + [len(subtokens)]))
        subtokens = torch.split(subtokens, split_idxs.tolist())
        subtokens = [x[:-1] if x[-1] == delimiter else x for x in subtokens[:-1]]  # filter delimiter after splitting
        tokens_to_subtokens = {idx + num_special_tokens: x.int().tolist() for idx, x in enumerate(subtokens)}
        return tokens_to_subtokens, index_character_map, iter

    def get_bigrams(self, flat_tokens_num, index_character_map, delimiter, end_subtoken) -> Tensor:
        end_token_idx = index_character_map.inverse[end_subtoken][0]
        # Get bigrams and eliminate all end of tokens and delimiters
        bigrams = torch.stack([flat_tokens_num[:-1], flat_tokens_num[1:]], dim=1)
        valid_idxs = bigrams[:, 0] != end_token_idx
        bigrams = bigrams[valid_idxs]  # eliminate bigrams beginning with end token
        non_delimiter_idxs = bigrams != delimiter
        # eliminate all bigrams containing delimiter token
        bigrams = bigrams[(non_delimiter_idxs[:, 0] & non_delimiter_idxs[:, 1])]
        bigram, counts = torch.unique(bigrams, return_counts=True, dim=0)
        if bigram.nelement() == 0:
            print("Could not find any bigrams, aborting...")
            return
        best_idx = torch.argmax(counts)  # get most frequent bigram
        best_bigram = bigram[best_idx]
        return best_bigram

    def merge_bigrams(self,
                      flat_tokens,
                      replace_bigram,
                      index_character_map,
                      end_subtoken,
                      num_special_tokens) -> Tuple[Tensor, dict]:
        end_token_idx = index_character_map.inverse[end_subtoken][0]
        unique_delimiter = -1
        intersection = (flat_tokens == replace_bigram[0]) & torch.roll(flat_tokens == replace_bigram[1], shifts=-1)
        idxs_replace_tensor = torch.where(intersection)[0]
        merged_bigram = np.array([len(index_character_map) + num_special_tokens])
        if replace_bigram[1] == end_token_idx:
            # append -1 as unique delimiter to merge sub-token sequences later
            merged_bigram = np.append(merged_bigram, unique_delimiter)
        merged_bigram_tensor = torch.tensor(merged_bigram, dtype=flat_tokens.dtype, device=flat_tokens.device)
        flat_tokens[idxs_replace_tensor] = merged_bigram_tensor[0]
        if len(merged_bigram_tensor) > 1:
            # in this case the length of flat tokens does not decrease
            flat_tokens[idxs_replace_tensor + 1] = merged_bigram_tensor[1]
        else:
            second_half_idxs_tensor = idxs_replace_tensor + 1
            mask = torch.ones(len(flat_tokens), dtype=torch.bool, device=second_half_idxs_tensor.device)
            mask[second_half_idxs_tensor] = False
            flat_tokens = flat_tokens[mask]
        first_half = index_character_map[int(replace_bigram[0].item())]
        second_half = index_character_map[int(replace_bigram[1].item())]
        index_character_map[merged_bigram[0]] = first_half + second_half  # update lookup dict with new bigram
        return flat_tokens, index_character_map