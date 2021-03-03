import re
from kge import Dataset
import torch
import time
import numpy as np
from torch.nn.utils.rnn import pad_sequence


# Algorithm:
# Stop if number of max. iteration or desired number of tokens are reached
# Count frequency of each word
# Split words into characters
# Count frequency of each consecutive byte-pair
# Find most frequent byte-pair
# Merge most frequent byte-pair into new token
# Replace individual character pairs by new, most frequent token
# output: vocabulary

class BytePairEncodingVocab:

    def __init__(self,
                 olp_dataset: Dataset,
                 iterations_entities,
                 iterations_relations
                 ):

        self.entity_sub_token_ids = None
        self.relation_sub_token_ids = None
        self.num_entity_sub_tokens = None
        self.num_relation_sub_tokens = None
        self.entity_sub_token_lookup = None
        self.relation_sub_token_lookup = None

        # run bpe and init vocab
        self.create_sub_token_vocabs(olp_dataset, iterations_entities, iterations_relations)

    def get_bigrams(self, vocab):
        split_vocab = [x.split(' ') for x in vocab]
        t_bi1 = time.time()
        bigrams = np.array(
            [' '.join(sub_token[i:i + 2]) for sub_token in split_vocab for i in range(len(sub_token) - 1)])
        t_bi2 = time.time()
        print("Text-bigram time: ", t_bi2 - t_bi1)
        return bigrams

    # TODO: improve performance
    # '</w>'.join(''.join(entity_token_vocab).split('</w>'))
    # if best_bigram[0] in x and best_idx[1] in
    def merge(self, best_bigram, vocab):
        first_half, second_half = best_bigram.split(" ")
        begin = '^' + first_half + '\s' + second_half + '\s'
        mid = '\s' + first_half + '\s' + second_half + '\s'
        end = '\s' + first_half + '\s' + second_half + '$'
        begin_end = '^' + first_half + '\s' + second_half + '$'
        pattern = re.compile('|'.join([begin, mid, end, begin_end]))
        vocab = [pattern.sub(' ' + best_bigram.replace(" ", "") + ' ', x).strip() for x in vocab]
        # entity_token_vocab = [x.replace(best_bigram, best_bigram.replace(" ", "")) for x in entity_token_vocab]
        # ' '.join(entity_token_vocab).replace(best_bigram, best_bigram.replace(" ", ""))
        return vocab

    def run_bpe(self, num_iters, token_vocab):
        t1 = time.time()
        for i in range(num_iters):
            bigrams = self.get_bigrams(token_vocab)
            if len(bigrams) == 0:
                print("Could not find any bigrams, aborting...")
                break
            # Count frequency of each consecutive byte-pair
            unique_bigrams, counts = np.unique(bigrams, return_counts=True)
            # Find most frequent byte-pair
            best_idx = np.argmax(counts)
            best_bigram = unique_bigrams[best_idx]
            # Merge most frequent byte-pairs in token vocabulary
            token_vocab = self.merge(best_bigram, token_vocab)

            # OPTIONAL:
            # count all sub-tokens in token vocabulary
            sub_token_vocab = np.unique(np.array(' '.join(token_vocab).split()))
            print("Number of sub-tokens: {} after iteration {}, most frequent bigram: {}".format(len(sub_token_vocab),
                                                                                                 i + 1, best_bigram))

            if i > 0 and (i + 1) % 10 == 0:
                print("Number of sub-tokens: {} after iteration {}".format(len(sub_token_vocab), i + 1))
                t2 = time.time()
                print("{}s per {} iterations".format(t2 - t1, 10))
                t1 = time.time()
        t2 = time.time()
        print("Total runtime: ", t2 - t1)
        return token_vocab

    def run_bpe_num(self, num_iterations, flat_sub_tokens, character_index_map):
        t1 = time.time()
        sub_token_dict = {}
        for x in range(num_iterations):
            # Step 2: Find most frequent bigram
            replace_bigram = self.get_bigrams_num(flat_sub_tokens, character_index_map)

            # Step 3: Merge most frequent bigram
            flat_sub_tokens, sub_token_dict = self.merge_bigrams_num_short(flat_sub_tokens, replace_bigram,
                                                                           character_index_map['</w>'])
            # flat_sub_tokens, sub_token_dict = self.merge_bigrams_num(flat_tokens_num, replace_bigram, character_index_map['</w>'], index_character_map, character_index_map)
        t2 = time.time()
        print("Bigram + merging time: {}".format(t2 - t1))
        return flat_sub_tokens, sub_token_dict

    def get_bigrams_num(self, flat_tokens_num, character_index_map):
        end_of_token = character_index_map["</w>"]
        t10 = time.time()
        end_token = [end_of_token, -1]  # -1 = unique delimiter for end of sub-token sequence
        t11 = time.time()
        t30 = time.time()

        # Get bigrams and eliminate all end of tokens and delimiters
        tb = torch.stack([flat_tokens_num[:-1], flat_tokens_num[1:]], dim=1) #.numpy()
        #elimination_idxs = torch.logical_or(torch.logical_or(tb[:, 1] == -1, tb[:, 0] == -1), )
        idxs = tb[:, 0] != end_of_token
        tb = tb[idxs]  # eliminate bigrams beginning with end token
        valid_idxs = tb != -1
        tb = tb[(valid_idxs[:, 0] & valid_idxs[:, 1])]  # eliminate all bigrams containing delimiter -1

        t12 = time.time()

        '''
        flat_tokens_num_np = flat_tokens_num.cpu().detach().numpy()
        bigrams = []
        #flat_tokens_num_np = flat_tokens_num
        for idx in range(len(flat_tokens_num_np) - 1):
            if flat_tokens_num_np[idx + 1] != -1 and flat_tokens_num_np[idx] not in end_token:
                bigrams.append(flat_tokens_num_np[idx: idx + 2].astype(int))
        bigrams_compare = torch.Tensor(bigrams).to(flat_tokens_num.device)
        #same = torch.equal(bigrams_new, bigrams_compare)
        same = torch.equal(tb, bigrams_compare)
        print("SAME: ", same)
        '''

        t13 = time.time()
        # bigrams_tensor = torch.stack(bigrams)
        bigrams_tensor = tb #bigrams_new #torch.Tensor(bigrams_new).to(flat_tokens_num.device)
        bigram, counts = torch.unique(bigrams_tensor, return_counts=True, dim=0)
        t14 = time.time()
        best_idx = torch.argmax(counts)
        t15 = time.time()
        best_bigram = bigram[best_idx]
        t16 = time.time()

        print("Bigram timings:")
        '''
        print(t11 - t10)
        print(t12 - t11)
        print(t13 - t12)
        print(t14 - t13)
        print(t15 - t14)
        print(t16 - t15)
        '''
        print("Bigram step     : ", t12 - t30)
        print("Bigram step loop: ", t13 - t12)
        print("Total: ", t16 - t10)

        # Todo: confirm that the results are the same as in text-based version for all iterations
        # best_bigram_ = [index_character_map[x] for x in best_bigram.cpu().detach().numpy()]
        return best_bigram

    def merge_bigrams_num_short(self, flat_tokens_num, replace_bigram, idx_end_token):
        # Step 3: merge best bigram in (sub-)token vocabulary
        t20 = time.time()
        intersection = (flat_tokens_num == replace_bigram[0]) & torch.roll(flat_tokens_num == replace_bigram[1], shifts=-1)
        idxs_replace_tensor = torch.where(intersection)[0]
        t21 = time.time()
        unique_sub_tokens = torch.unique(flat_tokens_num)
        t22 = time.time()
        merged_bigram = np.array([len(unique_sub_tokens[unique_sub_tokens > 0]) + 1])
        if replace_bigram[1] == idx_end_token:
            merged_bigram = np.append(merged_bigram,
                                      -1)  # append -1 as unique delimiter to split into sub-token sequences later
        t23 = time.time()
        # ---------- Torch solution -----------
        merged_bigram_tensor = torch.tensor(merged_bigram, dtype=flat_tokens_num.dtype, device=flat_tokens_num.device)
        t24 = time.time()
        flat_tokens_num[idxs_replace_tensor] = merged_bigram_tensor[0]
        t25 = time.time()
        # merged_bigram_tensor = torch.Tensor([39]).to(device)
        if len(merged_bigram_tensor) > 1:
            print("IF")
            flat_tokens_num[idxs_replace_tensor + 1] = merged_bigram_tensor[
                1]  # in this case the length of flat tokens does not decrease
        else:
            print("ELSE")
            second_half_idxs_tensor = idxs_replace_tensor + 1
            mask = torch.ones(len(flat_tokens_num), dtype=torch.bool, device=second_half_idxs_tensor.device)
            mask[second_half_idxs_tensor] = False
            flat_tokens_num = flat_tokens_num[mask]

        t26 = time.time()
        print("Merge step:")
        '''
        print(t21 - t20)
        print(t22 - t21)
        print(t23 - t22)
        print(t24 - t23)
        print(t25 - t24)
        print(t26 - t25)
        '''
        print("Total Merge: ", t26 - t20)

        return flat_tokens_num, {}

    def merge_bigrams_num(self, flat_tokens_num, replace_bigram, idx_end_token, index_character_map,
                          character_index_map):
        # Step 3: merge best bigram in (sub-)token vocabulary

        # pure torch solution
        idx_first_token = flat_tokens_num == replace_bigram[0]
        idx_second_token = flat_tokens_num == replace_bigram[1]
        intersection = idx_first_token & torch.roll(idx_second_token, shifts=-1)
        idxs_replace_tensor = torch.where(intersection)[0]

        # Identify if elimination of individual sub-tokens happens
        idxs_1 = torch.where(idx_first_token)[0]
        idxs_2 = torch.where(idx_second_token)[0]

        if len(idxs_replace_tensor) == len(idxs_1) and len(idxs_replace_tensor) == len(idxs_2):
            print("The bigram eliminates all its components: {}, {}{}".format(replace_bigram,
                                                                              index_character_map[replace_bigram[0]],
                                                                              index_character_map[replace_bigram[1]]))

        replace_bigram_ = [index_character_map[x] for x in replace_bigram.cpu().detach().numpy()]
        joined_sub_token = ''.join(replace_bigram_)

        end_of_token = character_index_map["</w>"]
        # Todo: # Assume that no sub-tokens are eliminated by their bigrams and fix later if this happens
        idx_to_merged_bigram = {len(index_character_map): joined_sub_token}
        merged_bigram_to_idx = {joined_sub_token: len(index_character_map)}

        if '</eot>' not in character_index_map:
            character_index_map.update({'</eot>': -1})
        if -1 not in index_character_map:
            index_character_map.update({-1: '</eot>'})

        character_index_map.update(merged_bigram_to_idx)
        index_character_map.update(idx_to_merged_bigram)

        merged_bigram = np.array([character_index_map[joined_sub_token]])
        if replace_bigram_[1] == '</w>':
            merged_bigram = np.append(merged_bigram,
                                      -1)  # append -1 as unique delimiter to split into sub-token sequences later

        # ---------- Torch solution -----------
        # merged_bigram_tensor = torch.tensor([inv_merged_bigram[joined_sub_token]], dtype=flat_tokens_num.dtype).to(device)
        merged_bigram_tensor = torch.tensor(merged_bigram, dtype=flat_tokens_num.dtype, device=flat_tokens_num.device)

        flat_tokens_num[idxs_replace_tensor] = merged_bigram_tensor[0]
        if len(merged_bigram_tensor) > 1:
            flat_tokens_num[idxs_replace_tensor + 1] = merged_bigram_tensor[
                1]  # in this case the length of flat tokens does not decrease
        else:
            second_half_idxs_tensor = idxs_replace_tensor + 1
            test = torch.ones(len(flat_tokens_num), dtype=torch.bool, device=second_half_idxs_tensor.device)
            test[second_half_idxs_tensor] = False
            flat_tokens_num = flat_tokens_num[test]

        # --------- a: Lookup string sub-tokens and rebuild updated flat tensor  ------------
        # Contra: Lookup is on CPU, because we use strings
        flat_tokens_num_np = flat_tokens_num.cpu().detach().numpy().astype(int)
        sub_tokens_str = np.array([index_character_map[x] for x in flat_tokens_num_np])
        unique_sub_tokens = np.unique(sub_tokens_str)
        if '</eot>' in unique_sub_tokens:
            unique_sub_tokens = np.delete(unique_sub_tokens, np.where(unique_sub_tokens == '</eot>'))
        # Questions:
        # Multi indexing via np array of indexes and strings instead of dict?

        str_to_index_map = {x: idx for idx, x in enumerate(unique_sub_tokens)}
        str_to_index_map.update({'</eot>': -1})
        # index_to_str_map = {v: k for k, v in str_to_index_map.items()}
        sub_tokens = [str_to_index_map[x] for x in sub_tokens_str]
        sub_tokens_updated = torch.Tensor(sub_tokens).to(flat_tokens_num.device)
        # -------- /a ------------

        return flat_tokens_num

    def create_sub_token_vocabs(self, olp_dataset, iterations_entities, iterations_relations):
        entity_tokens_str = olp_dataset._meta['entity_token_ids']
        relation_tokens_str = olp_dataset._meta['relation_token_ids']

        # get vocabulary, add stop word '</w>' at the end of each token
        entity_tokens = [' '.join(x) + ' </w>' for x in entity_tokens_str[4:]]
        total_entity_tokens = len(entity_tokens)
        total_entity_sub_tokens = len(np.unique(np.array(' '.join(entity_tokens).split())))
        print("Number of sub-tokens at the beginning: {}/{}".format(total_entity_sub_tokens, total_entity_tokens))
        entity_tokens_bpe = self.run_bpe(iterations_entities, entity_tokens)

        # -------------- Number-based / GPU Version -------------
        flat_tokens = np.concatenate([np.array(x.split()) for x in entity_tokens])
        unique_characters = np.unique(flat_tokens)
        character_index_map = {x: idx for idx, x in enumerate(unique_characters)}
        index_character_map = {v: k for k, v in character_index_map.items()}  # uniqueness of values!
        # Todo: test timing on olpbench
        d = dict([(y, x) for x, y in enumerate(sorted(set(flat_tokens)))])
        device = olp_dataset.config.get("job.device")
        flat_tokens_num = [character_index_map[x] for x in flat_tokens]
        flat_tokens_num_np = np.array([character_index_map[x] for x in flat_tokens])
        flat_sub_tokens = torch.Tensor(flat_tokens_num).to(device)
        flat_sub_tokens, sub_token_dict = self.run_bpe_num(iterations_entities, flat_sub_tokens, character_index_map)
        flat_sub_tokens_np = flat_sub_tokens.cpu().detach().numpy()
        split_idxs_new = np.where(np.logical_or(flat_sub_tokens_np == -1, flat_sub_tokens_np == 11))[0] + 1
        output = np.split(flat_sub_tokens_np, split_idxs_new)[:-1]
        output = [x[:-1].astype(int).tolist() if x[-1] == -1 else x.astype(int).tolist() for x in output]
        output_dict = {idx: x for idx, x in enumerate(output)}
        # /------------- Number-based / GPU Version -------------


        # ------------- Text-based version / CPU Version -------------
        entity_sub_tokens = np.unique(np.array(' '.join(entity_tokens_bpe).split()))
        self.num_entity_sub_tokens = len(entity_sub_tokens)
        self.entity_sub_token_ids = {x: idx for idx, x in enumerate(entity_sub_tokens)}

        self.entity_sub_token_lookup = {idx + 4: np.array([self.entity_sub_token_ids[y]
                                                           for y in x.split()])
                                        for idx, x in enumerate(entity_tokens_bpe)
                                        }
        self.entity_sub_token_lookup[3] = np.array([-1])  # keep end of tokens

        relation_tokens = [' '.join(x) + ' </w>' for x in relation_tokens_str[4:]]
        total_relation_tokens = len(relation_tokens)
        total_relation_sub_tokens = len(np.unique(np.array(' '.join(relation_tokens).split())))
        print("Number of sub-tokens at the beginning: {}/{}".format(total_relation_sub_tokens, total_relation_tokens))
        relation_tokens_bpe = self.run_bpe(iterations_relations, relation_tokens)
        self.relation_token_vocab = {relation_tokens_str[3:][idx]: x.split() for idx, x in
                                     enumerate(relation_tokens_bpe)}
        relation_sub_tokens = np.unique(np.array(' '.join(relation_tokens_bpe).split()))
        self.num_relation_sub_tokens = len(relation_sub_tokens)
        self.relation_sub_token_ids = {x: idx for idx, x in enumerate(relation_sub_tokens)}
        self.relation_sub_token_lookup = {idx + 4: np.array([self.relation_sub_token_ids[y]
                                                             for y in x.split()])
                                          for idx, x in enumerate(relation_tokens_bpe)
                                          }
        self.relation_sub_token_lookup[3] = np.array([-1])


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
            self.sub_token_lookup = bpe_vocab.entity_sub_token_lookup

        if "relation" in configuration_key:
            self.sub_token_lookup = bpe_vocab.relation_sub_token_lookup

        self.bpe_vocab = bpe_vocab

    def get_sub_tokens_from_tokens(self, token_indexes):
        import time
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
        # -------- Solution A -----------:
        # np.where(np.array([x.tolist() for x in sub_tokens]) == 3)
        sub_tokens_np = np.concatenate([self.sub_token_lookup[token_idx] for token_idx in token_idx_sequence])[:-1]
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
        # sts_np = [x.cpu().detach().numpy() for x in sub_token_tensor_single]         # Debug
        # Todo: check if this is correct for olpbench big
        output = (pad_sequence(sub_tokens_split).transpose(0, 1))[:, 1:]  # .cpu().detach().numpy()
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

        padded_sub_token_tensor = output
        return padded_sub_token_tensor