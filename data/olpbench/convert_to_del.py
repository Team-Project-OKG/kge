in_dir = "/mapped_to_ids/"
out_dir = ""

configs = [
    {
        "input_filename": "entity_id_map.txt",
        "output_filename": "entity_ids.del",
        "input_header_lines": 1,
        "mappings": [{"input_position": 1, "datatype": "int", "input_offset": -2}, {"input_position": 0, "datatype": "str"}]
    },
    {
        "input_filename": "relation_id_map.txt",
        "output_filename": "relation_ids.del",
        "input_header_lines": 1,
        "mappings": [{"input_position": 1, "datatype": "int", "input_offset": -2},
                     {"input_position": 0, "datatype": "str"}]
    },
    {
        "input_filename": "train_data_basic.txt",
        "output_filename": "train_basic.del",
        "input_header_lines": 0,
        "mappings": [{"input_position": 0, "datatype": "int", "input_offset": -2},
                     {"input_position": 1, "datatype": "int", "input_offset": -2},
                     {"input_position": 2, "datatype": "int", "input_offset": -2},
                     {"input_position": 3, "datatype": "int", "input_offset": -2},
                     {"input_position": 4, "datatype": "int", "input_offset": -2}]
    },
    {
        "input_filename": "train_data_simple.txt",
        "output_filename": "train_simple.del",
        "input_header_lines": 0,
        "mappings": [{"input_position": 0, "datatype": "int", "input_offset": -2},
                     {"input_position": 1, "datatype": "int", "input_offset": -2},
                     {"input_position": 2, "datatype": "int", "input_offset": -2},
                     {"input_position": 3, "datatype": "int", "input_offset": -2},
                     {"input_position": 4, "datatype": "int", "input_offset": -2}]
    },
    {
        "input_filename": "train_data_thorough.txt",
        "output_filename": "train_thorough.del",
        "input_header_lines": 0,
        "mappings": [{"input_position": 0, "datatype": "int", "input_offset": -2},
                     {"input_position": 1, "datatype": "int", "input_offset": -2},
                     {"input_position": 2, "datatype": "int", "input_offset": -2},
                     {"input_position": 3, "datatype": "int", "input_offset": -2},
                     {"input_position": 4, "datatype": "int", "input_offset": -2}]
    },
    {
        "input_filename": "test_data.txt",
        "output_filename": "test.del",
        "input_header_lines": 0,
        "mappings": [{"input_position": 0, "datatype": "int", "input_offset": -2},
                     {"input_position": 1, "datatype": "int", "input_offset": -2},
                     {"input_position": 2, "datatype": "int", "input_offset": -2},
                     {"input_position": 3, "datatype": "int", "input_offset": -2},
                     {"input_position": 4, "datatype": "int", "input_offset": -2}]
    },
    {
        "input_filename": "validation_data_all.txt",
        "output_filename": "validation_all.del",
        "input_header_lines": 0,
        "mappings": [{"input_position": 0, "datatype": "int", "input_offset": -2},
                     {"input_position": 1, "datatype": "int", "input_offset": -2},
                     {"input_position": 2, "datatype": "int", "input_offset": -2},
                     {"input_position": 3, "datatype": "int", "input_offset": -2},
                     {"input_position": 4, "datatype": "int", "input_offset": -2}]
    },
    {
        "input_filename": "validation_data_linked.txt",
        "output_filename": "validation_linked.del",
        "input_header_lines": 0,
        "mappings": [{"input_position": 0, "datatype": "int", "input_offset": -2},
                     {"input_position": 1, "datatype": "int", "input_offset": -2},
                     {"input_position": 2, "datatype": "int", "input_offset": -2},
                     {"input_position": 3, "datatype": "int", "input_offset": -2},
                     {"input_position": 4, "datatype": "int", "input_offset": -2}]
    },
    {
        "input_filename": "entity_id_tokens_ids_map.txt",
        "output_filename": "entity_id_tokens_ids_map.del",
        "input_header_lines": 1,
        "mappings": [{"input_position": 0, "datatype": "int", "input_offset": -2},
                     {"input_position": 1, "datatype": "int", "input_offset": 0}]
    },
    {
        "input_filename": "entity_token_id_map.txt",
        "output_filename": "entity_token_id_map.del",
        "input_header_lines": 1,
        "mappings": [{"input_position": 1, "datatype": "int", "input_offset": 0},
                     {"input_position": 0, "datatype": "str"}],
        "header": "0\t[unmapped]\n1\t[unseen]\n2\t[begin]\n3\t[end]\n"
    },
    {
        "input_filename": "relation_id_tokens_ids_map.txt",
        "output_filename": "relation_id_tokens_ids_map.del",
        "input_header_lines": 1,
        "mappings": [{"input_position": 0, "datatype": "int", "input_offset": -2},
                     {"input_position": 1, "datatype": "int", "input_offset": 0}]
    },
    {
        "input_filename": "relation_token_id_map.txt",
        "output_filename": "relation_token_id_map.del",
        "input_header_lines": 1,
        "mappings": [{"input_position": 1, "datatype": "int", "input_offset": 0},
                     {"input_position": 0, "datatype": "str"}],
        "header": "1\t[unseen]\n2\t[begin]\n3\t[end]\n"
    }

]

for config in configs:
    with open(in_dir + config["input_filename"], "r", encoding="utf8") as input_file:
        with open(out_dir + config["output_filename"], "w", encoding="utf8") as output_file:
            if "header" in config.keys():
                output_file.write(config["header"])
            line_count = 0
            for line in input_file:
                line_count += 1
                if line_count > config["input_header_lines"]:
                    inputs = line.strip().split("\t")
                    mapped_outputs = [" ".join([str(int(col_input) + mapping["input_offset"]) for col_input in inputs[mapping["input_position"]].split(" ")]) if mapping["datatype"] == "int" else inputs[mapping["input_position"]] for mapping in config["mappings"]]
                    output_file.write("\t".join(mapped_outputs) + "\n")
