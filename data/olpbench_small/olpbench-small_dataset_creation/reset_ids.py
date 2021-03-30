import os
import pandas as pd

files_directory = './'
reset_id_map_directory = './resetted_mapped_to_ids/'

token_id_map_header = pd.read_csv("./token_id_map_header.txt", delimiter="\t", names=["token", "token_id"])
entity_id_map = pd.read_csv("./entity_id_map.del", delimiter="\t", names=["entity", "entity_id"])
entity_id_tokens_ids_map = pd.read_csv("./entity_id_tokens_ids_map.del", delimiter="\t", names=["entity_id", "token_ids"])
entity_token_id_map = pd.concat([token_id_map_header, pd.read_csv("./entity_token_id_map.del", delimiter="\t", names=["token", "token_id"])], ignore_index=True)
relation_id_map = pd.read_csv("./relation_id_map.del", delimiter="\t", names=["relation", "relation_id"])
relation_id_tokens_ids_map = pd.read_csv("./relation_id_tokens_ids_map.del", delimiter="\t", names=["relation_id", "token_ids"])
relation_token_id_map = pd.concat([token_id_map_header, pd.read_csv("./relation_token_id_map.del", delimiter="\t", names=["token", "token_id"])], ignore_index=True)


new_entity_id_map = entity_id_map.reset_index().rename(columns={"entity_id": "old_entity_id", "index": "entity_id"}).set_index("old_entity_id")
new_relation_id_map = relation_id_map.reset_index().rename(columns={"relation_id": "old_relation_id", "index": "relation_id"}).set_index("old_relation_id")

new_entity_id_map_dict = new_entity_id_map["entity_id"].to_dict()
new_relation_id_map_dict = new_relation_id_map["relation_id"].to_dict()


new_entity_token_id_map = entity_token_id_map.reset_index().rename(columns={"token_id": "old_token_id", "index": "token_id"}).set_index("old_token_id")
new_relation_token_id_map = relation_token_id_map.reset_index().rename(columns={"token_id": "old_token_id", "index": "token_id"}).set_index("old_token_id")

new_entity_token_id_map_dict = new_entity_token_id_map["token_id"].to_dict()
new_relation_token_id_map_dict = new_relation_token_id_map["token_id"].to_dict()



new_entity_id_tokens_ids_map = entity_id_tokens_ids_map.copy()

new_entity_id_tokens_ids_map["entity_id"] = entity_id_tokens_ids_map["entity_id"].replace(new_entity_id_map_dict)

new_entity_id_tokens_ids_map["token_ids"] = new_entity_id_tokens_ids_map["token_ids"].apply(lambda token_ids: " ".join([str(new_entity_token_id_map_dict.get(int(splitted), None)) for splitted in token_ids.split(" ")]))


new_relation_id_tokens_ids_map = relation_id_tokens_ids_map.copy()

new_relation_id_tokens_ids_map["relation_id"] = relation_id_tokens_ids_map["relation_id"].replace(new_relation_id_map_dict)

new_relation_id_tokens_ids_map["token_ids"] = new_relation_id_tokens_ids_map["token_ids"].apply(lambda token_ids: " ".join([str(new_relation_token_id_map_dict.get(int(splitted), None)) for splitted in token_ids.split(" ")]))



mapping_files = [f for f in os.listdir(files_directory) if any([x in f for x in ["test", "train", "validation"]])]

for file_name in mapping_files:
    file_df = pd.read_csv(files_directory+file_name, delimiter="\t", names=["subject_id", "relation_id", "object_id", "alternative_subjects", "alternative_objects"])
    file_df["subject_id"] = file_df["subject_id"].replace(new_entity_id_map_dict)
    file_df["object_id"] = file_df["object_id"].replace(new_entity_id_map_dict)
    file_df["relation_id"] = file_df["relation_id"].replace(new_relation_id_map_dict)
    file_df["alternative_subjects"] = file_df["alternative_subjects"].apply(lambda token_ids: " ".join([str(new_entity_id_map_dict.get(int(splitted), None)) for splitted in str(token_ids).split(" ")]))
    file_df["alternative_objects"] = file_df["alternative_objects"].apply(lambda token_ids: " ".join([str(new_entity_id_map_dict.get(int(splitted), None)) for splitted in str(token_ids).split(" ")]))
    file_df.to_csv(reset_id_map_directory+file_name, sep="\t", header=False, index=False)
    print(file_name + " done")


new_entity_id_map.to_csv(reset_id_map_directory+"entity_id_map.del", sep="\t", header=False, index=False)
new_entity_id_tokens_ids_map.to_csv(reset_id_map_directory+"entity_id_tokens_ids_map.del", sep="\t", header=False, index=False)
new_entity_token_id_map.to_csv(reset_id_map_directory+"entity_token_id_map.del", sep="\t", header=False, index=False)
new_relation_id_map.to_csv(reset_id_map_directory+"relation_id_map.del", sep="\t", header=False, index=False)
new_relation_id_tokens_ids_map.to_csv(reset_id_map_directory+"relation_id_tokens_ids_map.del", sep="\t", header=False, index=False)
new_relation_token_id_map.to_csv(reset_id_map_directory+"relation_token_id_map.del", sep="\t", header=False, index=False)


print("done")