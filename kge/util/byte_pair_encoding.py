from kge import Dataset
import torch
import time
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class Bidict(dict):
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
        return super(Bidict, self).copy()


class BytePairEncodingVocab:
    """
    Algorithm:
    Stop if number of max. iteration or desired number of tokens are reached
        Count frequency of each word
        Split words into characters
        Count frequency of each consecutive byte-pair
        Find most frequent byte-pair
        Merge most frequent byte-pair into new token
        Replace individual character pairs by new, most frequent token
        output: vocabulary of token index -> sub-token indexes
    """

    def __init__(self,
                 olp_dataset: Dataset,
                 iterations_entities,
                 iterations_relations
                 ):

        self.ent_sub_token_ids = None
        self.rel_sub_token_ids = None
        self.num_ent_sub_tokens = None
        self.num_rel_sub_tokens = None
        self.ent_sub_token_lookup = None
        self.rel_sub_token_lookup = None
        # run bpe and init vocab
        self.create_sub_token_vocabs(olp_dataset, iterations_entities, iterations_relations)

    def run_bpe_num(self, iterations, sub_tokens, index_character_map, end_sub_token, delimiter, num_special_tokens):
        for x in range(iterations):
            t1 = time.time()
            print("Iteration: ", x)
            # Step 2: Find most frequent bigram
            replace_bigram = self.get_bigrams(sub_tokens, index_character_map, delimiter, end_sub_token)
            if replace_bigram == None:
                break
            # Step 3: Merge most frequent bigram
            sub_tokens, index_character_map = self.merge_bigrams(sub_tokens, replace_bigram, index_character_map,
                                                                 end_sub_token, num_special_tokens)
            print("Merge bigram: ({} {})".format(index_character_map[int(replace_bigram[0].item())], index_character_map[int(replace_bigram[1].item())]))
            t2 = time.time()
            print("Merging + Bigrams: Time for one iteration: {}".format(t2 - t1))
        # keep only unique sub-tokens occuring in sub-token sequences after BPE
        unique_sub_tokens = torch.unique(sub_tokens[sub_tokens >= 0])
        index_character_map = Bidict({int(x.item()): index_character_map[int(x.item())] for x in unique_sub_tokens})
        if end_sub_token in index_character_map.inverse: # split at end_sub_token idx and delimiter
            end_sub_token_idx = index_character_map.inverse[end_sub_token][0]
            end_idxs = torch.where(torch.logical_or(sub_tokens == delimiter, sub_tokens == end_sub_token_idx))[0] + 1
        else:  # split only at delimiter
            end_idxs = torch.where(sub_tokens == delimiter)[0] + 1
        split_idxs = np.diff(([0] + end_idxs.tolist() + [len(sub_tokens)]))
        sub_tokens = torch.split(sub_tokens, split_idxs.tolist())
        sub_tokens = [x[:-1] if x[-1] == delimiter else x for x in sub_tokens[:-1]]  # filter delimiter after splitting
        tokens_to_sub_tokens = {idx + num_special_tokens: x.int().tolist() for idx, x in enumerate(sub_tokens)}
        return tokens_to_sub_tokens, index_character_map

    def get_bigrams(self, flat_tokens_num, index_character_map, delimiter, end_sub_token):
        end_token_idx = index_character_map.inverse[end_sub_token][0]
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
            return None
        best_idx = torch.argmax(counts)  # get most frequent bigram
        best_bigram = bigram[best_idx]
        return best_bigram

    def merge_bigrams(self, flat_tokens_num, replace_bigram, index_character_map, end_sub_token, num_special_tokens):
        end_token_idx = index_character_map.inverse[end_sub_token][0]
        unique_delimiter = -1
        intersection = (flat_tokens_num == replace_bigram[0]) & torch.roll(flat_tokens_num == replace_bigram[1], shifts=-1)
        idxs_replace_tensor = torch.where(intersection)[0]
        merged_bigram = np.array([len(index_character_map) +  num_special_tokens])
        if replace_bigram[1] == end_token_idx:
            # append -1 as unique delimiter to merge sub-token sequences later
            merged_bigram = np.append(merged_bigram, unique_delimiter)
        merged_bigram_tensor = torch.tensor(merged_bigram, dtype=flat_tokens_num.dtype, device=flat_tokens_num.device)
        flat_tokens_num[idxs_replace_tensor] = merged_bigram_tensor[0]
        if len(merged_bigram_tensor) > 1:
            # in this case the length of flat tokens does not decrease
            flat_tokens_num[idxs_replace_tensor + 1] = merged_bigram_tensor[1]
        else:
            second_half_idxs_tensor = idxs_replace_tensor + 1
            mask = torch.ones(len(flat_tokens_num), dtype=torch.bool, device=second_half_idxs_tensor.device)
            mask[second_half_idxs_tensor] = False
            flat_tokens_num = flat_tokens_num[mask]
        first_half = index_character_map[int(replace_bigram[0].item())]
        second_half = index_character_map[int(replace_bigram[1].item())]
        index_character_map[merged_bigram[0]] = first_half + second_half  # update lookup dict with new bigram
        return flat_tokens_num, index_character_map

    def create_sub_token_vocabs(self, olp_dataset, iterations_entities, iterations_relations):
        entity_tokens_str = olp_dataset._meta['entity_token_ids']
        relation_tokens_str = olp_dataset._meta['relation_token_ids']
        delimiter = -1
        end_sub_token = '</w>'
        device = olp_dataset.config.get("job.device")
        num_special_tokens = 4  # '[unmapped]', '[unseen]', '[begin]', '[end]'
        # exclude these tokens from byte-pair encoding
        special_tokens = {idx: token for idx, token in enumerate(entity_tokens_str[:num_special_tokens])}

        # Byte-Pair-Encoding for entities
        # get vocabulary, add stop word '</w>' at the end of each token
        entity_tokens = [' '.join(x) + ' ' + end_sub_token for x in entity_tokens_str[num_special_tokens:]]
        tokens_1d = np.concatenate([np.array(x.split()) for x in entity_tokens])
        unique_characters = np.unique(tokens_1d)
        # use a bidirectional map to update lookups and their inverse simultaneously
        # map index to sub-tokens and inverse
        index_character_map_ent = Bidict({idx + num_special_tokens: x for idx, x in enumerate(unique_characters)})
        sub_tokens_ent = torch.Tensor([index_character_map_ent.inverse[x][0] for x in tokens_1d]).to(device)
        self.ent_sub_token_lookup, index_character_map_ent = self.run_bpe_num(iterations_entities, sub_tokens_ent,
                                                                               index_character_map_ent, end_sub_token,
                                                                               delimiter, num_special_tokens)
        self.ent_sub_token_lookup = {**{key: [key] for key in special_tokens}, **self.ent_sub_token_lookup}
        self.ent_sub_token_ids = {**special_tokens, **index_character_map_ent.get_dict()}  # add special tokens again
        self.num_ent_sub_tokens = len(self.ent_sub_token_ids)
        #output_str = [[index_character_map[y.item()] for y in x] for x in output]
        #output_dict_str = {idx: x for idx, x in enumerate(output_str)}
        print(" ------------------------------------------------ \n")

        # Byte-Pair-Encoding for relations
        # get vocabulary, add stop word '</w>' at the end of each token
        relation_tokens = [' '.join(x) + ' ' + end_sub_token for x in relation_tokens_str[num_special_tokens:]]
        relation_tokens_1d = np.concatenate([np.array(x.split()) for x in relation_tokens])
        rel_unique_chars = np.unique(relation_tokens_1d)
        index_character_map_rel = Bidict({idx + num_special_tokens: x for idx, x in enumerate(rel_unique_chars)})
        sub_tokens_rel = torch.Tensor([index_character_map_rel.inverse[x][0] for x in relation_tokens_1d]).to(device)
        self.rel_sub_token_lookup, index_character_map_rel = self.run_bpe_num(iterations_relations, sub_tokens_rel,
                                                                           index_character_map_rel, end_sub_token,
                                                                           delimiter, num_special_tokens)
        self.rel_sub_token_lookup = {**{key: [key] for key in special_tokens}, **self.rel_sub_token_lookup}
        self.rel_sub_token_ids = self.ent_sub_token_ids = {**special_tokens, **index_character_map_rel.get_dict()}
        self.num_rel_sub_tokens = len(self.rel_sub_token_ids)
        #output_str = [[index_character_map[y.item()] for y in x] for x in output]
        #output_dict_str = {idx: x for idx, x in enumerate(output_str)}
        x = 0


# Todo: encode unseen words
class BPESubTokenEmbedder:
    """Byte-Pair Encoding Sub-Token Embedder takes in token index sequence.
       Splits tokens into sub-tokens, embeds sub-tokens into tensors and
       returns sub-token embeddings.
    """

    def __init__(self,
                 bpe_vocab: BytePairEncodingVocab,
                 configuration_key: str,
                 ):
        if "entity" in configuration_key:
            self.sub_token_lookup = bpe_vocab.ent_sub_token_lookup
        if "relation" in configuration_key:
            self.sub_token_lookup = bpe_vocab.rel_sub_token_lookup

        # Todo: configuration key is None???
        self.bpe_vocab = bpe_vocab

    import time
    def get_sub_tokens_from_tokens(self, token_indexes):
        t1 = time.time()
        # Create sub-word/token embeddings
        token_indexes = token_indexes[:, 1:]
        # entity_token_str idx 2 -> 2 (end), 3 -> [x, x , x]
        # token_idx_sequence: 2 x x x 3 0 0 0 -> lookup -> 1d sub-token sequence -> split at 3 -> tensor with padding
        t2 = time.time()
        # convert token to sub-tokens (dict lookup enforces cpu)
        # token_indexes[torch.where(token_indexes == 3)] = -1   # -1 is unique end of sequence token
        token_idx_sequence = token_indexes[token_indexes > 2].cpu().detach().numpy()  # split at 3
        t3 = time.time()
        # [x for x in np.split(sub_tokens_np, offsets_np)[:-1] if x[-1] == 3]

        # Todo: fix this
        self.sub_token_lookup[3] = [-1]

        # -------- Solution A -----------:
        # np.where(np.array([x.tolist() for x in sub_tokens]) == 3)
        sub_tokens_np = np.concatenate([self.sub_token_lookup[token_idx] for token_idx in token_idx_sequence])[:-1] # todo: token_idx.item()
        t4 = time.time()
        sub_tokens_np = np.insert(sub_tokens_np, 0, -1)  # insert 3 at idx 0 -> cut token 3 at last step
        t5 = time.time()
        offsets_np = np.where(sub_tokens_np == -1)[0]  # + 1 # solution for self.sub_token_lookup[3] = np.array([3])
        t6 = time.time()
        # Todo: check if we can remove  [:-1] from sub_tokens_np and delete appending len(sub_tokens_np)
        offset_by_len = np.diff(np.append(np.insert(offsets_np, 0, 0),
                                          len(sub_tokens_np)))  # Todo: delete inserting -1 -> should make no difference
        t7 = time.time()
        sub_token_tensor = torch.tensor(sub_tokens_np, device=token_indexes.device)
        t8 = time.time()
        sub_tokens_split = torch.split(sub_token_tensor, offset_by_len.tolist())[1:]
        t9 = time.time()
        # Todo: check if this is correct for olpbench big
        padded_sub_tokens = (pad_sequence(sub_tokens_split).transpose(0, 1))[:, 1:]  # .cpu().detach().numpy()
        t10 = time.time()
        # Todo: option to keep or drop start and end of sub-token sequence
        # if keep_start_end_token
        # o1 = output.cpu().detach().numpy()
        # print("Total: ", t10-t1)
        '''
        print("1,2", t2 - t1)
        print("2,3", t3 - t2)
        print("3,4", t4 - t3)
        print("4,5", t5 - t4)
        print("5,6", t6 - t5)
        print("6,7", t7 - t6)
        print("7,8", t8 - t7)
        print("8,9", t9 - t8)
        print("9,10", t10 - t9)
        '''
        # -------- /Solution A -----------:
        t2 = time.time()

        return padded_sub_tokens