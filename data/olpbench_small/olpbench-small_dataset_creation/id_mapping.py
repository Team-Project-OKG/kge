import os

# 1. loop through all datasets: get list of all entity id's and relation id's
# 2. map from ent/rel id -> ent/rel string
# 3. map from ent/rel id -> token ids
# 4. map from token_ids to token strings

id_map_directory = './olpbench/mapped_to_ids/'
data_directory = './olpbench-small/'

mapping_files = [f for f in os.listdir(id_map_directory) if "id" in f]
data_files = [f for f in os.listdir(data_directory) if f.startswith("train") or f.startswith("valid") or f.startswith("test")]

# 1. loop through all datasets: get list of all entity id's and relation id's
# A: From files
observed_ids = {"entities.txt": [], "relations.txt": []}
for filename in observed_ids.keys():
    with open(data_directory + filename, "r") as f:
        observed_ids[filename] = set([int(line.rstrip().split('\n')[0]) for line in f.readlines()[1:]])
print("Total entities: {}, total relations: {}".format(len(observed_ids["entities.txt"]), len(observed_ids["relations.txt"])))


'''
# B: From data
observed_entity_ids = set()
observed_relation_ids = set()
for filename in data_files:
    with open(data_directory + filename, "r") as f:
        data = [line.rstrip().split('\t') for line in f.readlines()]
        for line in data:
            relation_id = line[0].split()[1]
            observed_relation_ids.add(relation_id)  # add observed relation to set
            entity_ids = line[0].split()[0:3:2]
            observed_entity_ids = observed_entity_ids.union(set(entity_ids))
            alternative_entity_ids = [x.split() for x in line[1:3]]
            for x in alternative_entity_ids:
                observed_entity_ids = observed_entity_ids.union(set(x))
sorted_ent = set([int(x) for x in observed_entity_ids])
sorted_rel = set([int(x) for x in observed_relation_ids])
print("Total entities: {}, total relations: {}".format(len(observed_entity_ids), len(observed_relation_ids)))
'''

# 2. map from ent/rel id -> ent/rel string
id_map = {"entity_id_map.txt": {}, "relation_id_map.txt": {}}
for filename in id_map.keys():
    with open(id_map_directory + filename, "r") as f:
        for idx, line in enumerate(f):
            if idx > 0:
                line = line.rstrip().split("\t")[0:2]
                if filename == "entity_id_map.txt" and int(line[1]) in observed_ids["entities.txt"] \
                        or filename == "relation_id_map.txt" and int(line[1]) in observed_ids["relations.txt"]:
                    id_map[filename][int(line[1])] = line[0]

# 3. map from ent/rel id -> token ids
entity_tokens = set()
relation_tokens = set()
max_token_ents = 0
max_token_rels = 0
token_id_map = {"entity_id_tokens_ids_map.txt": {}, "relation_id_tokens_ids_map.txt": {}}
for filename in token_id_map.keys():
    with open(id_map_directory + filename, "r") as f:
        for idx, line in enumerate(f):
            if idx > 0:
                ent_rel_id, token_ids = line.strip("\n").split("\t")
                if int(ent_rel_id) in observed_ids["entities.txt"] and filename == "entity_id_tokens_ids_map.txt"\
                        or int(ent_rel_id) in observed_ids["relations.txt"] and filename == "relation_id_tokens_ids_map.txt":
                    token_ids = token_ids[2:-2]   # 2 marks beginning and 3 marks end of tokens -> remove both
                    token_id_map[filename][ent_rel_id] = token_ids
                    if filename == "entity_id_tokens_ids_map.txt":
                        entity_tokens = entity_tokens.union(set([int(x) for x in token_ids.split()]))
                        max_token_ents = max(max_token_ents, len(token_ids.split()) + 2)
                    elif filename == "relation_id_tokens_ids_map.txt":
                        relation_tokens = relation_tokens.union(set([int(x) for x in token_ids.split()]))
                        max_token_rels = max(max_token_rels, len(token_ids.split()) + 2)

# 4. map from token_ids to token strings
token_id_string_map = {"entity_token_id_map.txt": {}, "relation_token_id_map.txt": {}}
for filename in token_id_string_map.keys():
    with open(id_map_directory + filename, "r") as f:
        for idx, line in enumerate(f):
            if idx > 0:
                line = line.strip().split("\t")[0:2]
                token_id_string_map[filename][int(line[1])] = line[0]

token_ids_to_strings = {"entity_token_id_map.txt": {x: token_id_string_map["entity_token_id_map.txt"][x] for x in entity_tokens},
            "relation_token_id_map.txt": {x: token_id_string_map["relation_token_id_map.txt"][x] for x in relation_tokens}}

# Write into files
for filename, e_r_ids in id_map.items():
    file = open(filename[:-4] + ".del", "w+")
    for id_, tokens in e_r_ids.items():
        file.write(tokens + "\t" + str(id_) + "\n")
    file.close()

for filename, ent_to_tokens in token_id_map.items():
    file = open(filename[:-4] + ".del", "w+")
    for e_r_id, tokens in ent_to_tokens.items():
        file.write(str(e_r_id) + "\t" + "2 " + tokens + " 3\n")
    file.close()

for filename, token_to_str in token_ids_to_strings.items():
    file = open(filename[:-4] + ".del", "w+")
    for token_id, token_str in token_to_str.items():
        file.write(token_str + "\t" + str(token_id) + "\n")
    file.close()

file = open("dataset_infos.txt", "w+")
file.write("num_tokens_entities: " + str(len(token_ids_to_strings["entity_token_id_map.txt"])) + "\n")
file.write("num_tokens_relations: " + str(len(token_ids_to_strings["relation_token_id_map.txt"])) + "\n")
file.write("max_tokens_per_entity: " + str(max_token_ents) + "\n")
file.write("max_tokens_per_relation: " + str(max_token_rels) + "\n")
file.close()

print("Done")
