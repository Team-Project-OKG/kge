import datetime
import logging
import os
import time
from typing import Tuple

import pandas as pd
import yaml

from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
from avro.schema import parse

from kge.misc import kge_base_dir


def _read_yaml() -> dict:
    """
    Read config sentences_from_opiec.yaml.
    """
    with open(os.path.join(kge_base_dir(), "kge", "util", "sentences_from_opiec.yaml")) as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)


def _olpbench_triples_to_set(
        directory: str,
        filename: str
) -> set:
    """
    Read olpbench quintuples file that is using strings instead of IDs.
    """
    data = pd.read_csv(
        os.path.join(directory, filename), sep="\t", header=None, usecols=range(0, 3)
    )
    triples = set()
    triples.update(list(data.itertuples(index=False, name=None)))
    return triples


def _read_olpbench_triples(
        config: dict
) -> Tuple[set, set]:
    """
    Read train, valid and test olpbench quintuples files that are using strings instead of IDs.
    """
    logging.info("Start reading OLPBench triples.")
    directory = os.path.join(kge_base_dir(), "data", config["dataset"]["dir"])
    train = _olpbench_triples_to_set(directory, config["dataset"]["train_filename"])
    valid_test = _olpbench_triples_to_set(directory, config["dataset"]["valid_filename"])
    valid_test.update(_olpbench_triples_to_set(directory, config["dataset"]["test_filename"]))
    logging.info("Finished reading OLPBench triples.")
    return train, valid_test


def _find_matches_in_opiec(
        config: dict,
        train_triples: set,
        valid_test_triples: set
) -> set:
    """
    Read OPIEC files one after the other and search for matches in the given train, valid and test triples. Each
    triple being found is a separate match. Extracts information about subject, relation, object and their NER tags,
    and the original sentence from Wikipedia.
    Preliminarily writes matches every 250.000 hits to avoid the process being killed because of resource usage.
    Return matched sentences in valid and test triples to filter from matched triples afterwards.
    """
    matched_sentences = list()
    matched_sentences_dict = dict()
    train_valid_sentences = set()
    total_matches = 0
    file_number = 0
    triples_dir = config["opiec"]["triples_dir"]
    folder = os.path.join(kge_base_dir(), "opiec")
    AVRO_SCHEMA_FILE = os.path.join(folder, config["opiec"]["schema_filename"])
    files_in_folder = sorted(os.listdir(os.path.join(folder, triples_dir)))
    logging.info(f"Found {len(files_in_folder)} files in folder {triples_dir}.")
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, triples_dir, avro_filename)
            logging.info(f"Start looking for matches in file {avro_filename}.")
            start_timestamp = time.time()
            reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
            for triple in reader:
                # obtain subject, relation and object, adopted from Samuel Broscheit OLPBench creation code
                # https://github.com/samuelbroscheit/open_knowledge_graph_embeddings/blob/1ce37a4261a37e357a0f4dac3ee130ff11cbea4e/preprocessing/process_avro.py#L16
                # seemingly most efficient version from various timed options
                subject_lc = " ".join([w['word'] if 'QUANT' not in w['word']
                                       else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities']
                                       else w['word'] for w in
                                       sorted(triple['subject'] + triple['dropped_words_subject'],
                                              key=lambda x: x['index'])]
                                      ).lower()
                relation_lc = " ".join([w['word'] if 'QUANT' not in w['word']
                                        else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities']
                                        else w['word'] for w in
                                        sorted(triple['relation'] + triple['dropped_words_relation'],
                                               key=lambda x: x['index'])]
                                       ).lower()
                object_lc = " ".join([w['word'] if 'QUANT' not in w['word']
                                      else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities']
                                      else w['word'] for w in
                                      sorted(triple['object'] + triple['dropped_words_object'],
                                             key=lambda x: x['index'])]
                                     ).lower()

                if (subject_lc, relation_lc, object_lc) in train_triples:
                    total_matches += 1
                    # obtain NER tags
                    subject_ner = {w['ner'] for w in triple['subject'] + triple['dropped_words_subject']
                                   if w['ner'] != 'O'}
                    if len(subject_ner) == 0:
                        subject_ner.add('O')
                    relation_ner = {w['ner'] for w in triple['relation'] + triple['dropped_words_relation']
                                    if w['ner'] != 'O'}
                    if len(relation_ner) == 0:
                        relation_ner.add('O')
                    object_ner = {w['ner'] for w in triple['object'] + triple['dropped_words_object']
                                  if w['ner'] != 'O'}
                    if len(object_ner) == 0:
                        object_ner.add('O')
                    # obtain sentence
                    sentence = " ".join(
                        [token["word"] for token in triple["sentence_linked"]["tokens"]
                         if len(token["word"]) > 1 or token["word"].isalpha()]
                    ).lower()
                    matched_sentences.append({
                        "subject": {
                            "text": subject_lc,
                            "ner": list(subject_ner)
                        },
                        "relation": {
                            "text": relation_lc,
                            "ner": list(relation_ner)
                        },
                        "object": {
                            "text": object_lc,
                            "ner": list(object_ner)
                        },
                        "sentence": sentence
                    })
                    # add sentence to dictionary
                    try:
                        matched_sentences_dict[sentence].append(len(matched_sentences) - 1)
                    except KeyError:
                        matched_sentences_dict[sentence] = [len(matched_sentences) - 1]
                if (subject_lc, relation_lc, object_lc) in valid_test_triples:
                    # obtain sentence
                    sentence = " ".join(
                        [token["word"] for token in triple["sentence_linked"]["tokens"]
                         if len(token["word"]) > 1 or token["word"].isalpha()]
                    ).lower()
                    train_valid_sentences.add(sentence)
            logging.info(f"Finished searching for matches in {avro_filename} "
                  f"(runtime: {(time.time() - start_timestamp):.2f}s; total matches: {total_matches}).")
            reader.close()
        while len(matched_sentences) >= config["write_every"]:
            _write_matches_to_avro(
                config["output_matches"]["dir"],
                config["output_matches"]["schema_filename"],
                f"matched_triples_{file_number:03d}.avro",
                matched_sentences[0:config["write_every"]]
            )
            file_number += 1
            matched_sentences = matched_sentences[config["write_every"]:]
    # write final matches to file
    _write_matches_to_avro(
        config["output_matches"]["dir"],
        config["output_matches"]["schema_filename"],
        f"matched_triples_{file_number:03d}.avro",
        matched_sentences,
    )
    return train_valid_sentences


def _write_matches_to_avro(
        target_dir: str,
        schema_filename: str,
        filename: str,
        matched_sentences: list
):
    """
    Write a list of matches to an avro file as specified.
    """
    folder = os.path.join(kge_base_dir(), "opiec")

    AVRO_SCHEMA_FILE = os.path.join(folder, schema_filename)
    schema = parse(open(AVRO_SCHEMA_FILE, "rb").read())
    AVRO_FILE = os.path.join(folder, target_dir, filename)

    writer = DataFileWriter(open(AVRO_FILE, "wb"), DatumWriter(), schema)
    for matched_sentence in matched_sentences:
        writer.append(matched_sentence)
    writer.close()

    logging.info(f"{len(matched_sentences)} matched sentences successfully written to avro file {filename}.")


def _check_train_valid_sentences(
        config: dict,
        valid_test_sentences: set
):
    """
    Check if the sentence of matched triples in training data equals the sentence of matched triples of validation and
    test data. Remove the matched triples of the training data to avoid data leakage.
    Overwrites each file of matched triples in training data with the remaining matches.
    """
    folder = os.path.join(kge_base_dir(), "opiec")
    output_dir = config["output_matches"]["dir"]
    AVRO_SCHEMA_FILE = os.path.join(folder, config["output_matches"]["schema_filename"])
    files_in_folder = sorted(os.listdir(os.path.join(folder, output_dir)))
    logging.info(f"Found {len(files_in_folder)} files in folder {output_dir}.")
    total_fine_matches = 0
    total_original_matches = 0
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, output_dir, avro_filename)
            start_timestamp = time.time()
            reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
            fine_matches = list()
            original_matches = 0
            for triple in reader:
                original_matches += 1
                if not triple["sentence"] in valid_test_sentences:
                    fine_matches.append(triple)
            reader.close()
            logging.info(f"{len(fine_matches)} of {original_matches} matches remain after filtering validation and "
                  f"training sentences (runtime: {(time.time() - start_timestamp):.2f}s).")
            _write_matches_to_avro(
                config["output_matches"]["dir"],
                config["output_matches"]["schema_filename"],
                avro_filename,
                fine_matches
            )
            total_fine_matches += len(fine_matches)
            total_original_matches += original_matches
    logging.info(f"In total, {total_fine_matches} of {total_original_matches} matches remain after filtering validation and "
          f"training sentences")


def _restructure_sentences(
        config: dict
):
    """
    Create new set of avro files from matched triples in training data. Instead of each entry being linked to a triple,
    each entry is linked to a sentence. For each sentence, a list of subjects, relations and objects and their
    respective NER tags of triples that were extracted from that sentence is saved.
    """
    sentence_dict = dict()
    folder = os.path.join(kge_base_dir(), "opiec")
    triples_output_dir = config["output_matches"]["dir"]
    AVRO_SCHEMA_FILE = os.path.join(folder, config["output_matches"]["schema_filename"])
    files_in_folder = sorted(os.listdir(os.path.join(folder, triples_output_dir)))
    logging.info(f"Found {len(files_in_folder)} files in folder {triples_output_dir}.")
    logging.info(f"Start reading sentencens from files...")
    sentences = set()
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, triples_output_dir, avro_filename)
            start_timestamp = time.time()
            reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
            for triple in reader:
                sentences.add(triple["sentence"])
            reader.close()
            logging.info(f"Finished extracting sentences from file {avro_filename} "
                  f"(runtime: {(time.time() - start_timestamp):.2f}s, total sentences extracted: {len(sentences)})")

    # create temporary txt file with sentences for everything that is not in the first iteration
    logging.info("Start writing temporary txt file with sentences...")
    start_timestamp = time.time()
    sentences = list(sentences)
    pack_size = config["output_sentences"]["pack_size"]
    tmp_filename = os.path.join(folder, "sentences_tmp.txt")
    with open(tmp_filename, "w") as writer:
        for sentence in sentences[pack_size:]:
            writer.write(f"{sentence}\n")
        writer.close()
    sentences = sentences[0:config["output_sentences"]["pack_size"]]
    logging.info(f"Finished creating temporary dump file (runtime: {(time.time() - start_timestamp):.2f}s).")

    # extract triple information for these sentences
    tmp_file = open(tmp_filename, "r")
    file_number = 0
    while True:
        tmp_sentence = tmp_file.readline().strip()
        if len(sentences) % pack_size == 0 or not tmp_sentence:
            # if pack size is reached, check all avro files for triples information and then write it to avro files
            # before loading additional data
            sentence_dict = {
                sentence: {
                    'sentence': sentence,
                    'subjects': {
                        'texts': [],
                        'ner_lists': []
                    },
                    'relations': {
                        'texts': [],
                        'ner_lists': []
                    },
                    'objects': {
                        'texts': [],
                        'ner_lists': []
                    }
                }
                for sentence in sentences
            }
            del sentences
            for avro_filename in files_in_folder:
                if avro_filename.endswith(".avro"):
                    AVRO_FILE = os.path.join(folder, triples_output_dir, avro_filename)
                    start_timestamp = time.time()
                    reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
                    for triple in reader:
                        sentence = triple["sentence"]
                        if sentence in sentence_dict:
                            dict_value = sentence_dict[sentence]
                            if triple["subject"]["text"] not in dict_value["subjects"]["texts"]:
                                dict_value["subjects"]["texts"].append(triple["subject"]["text"])
                                dict_value["subjects"]["ner_lists"].append(triple["subject"]["ner"])
                            if triple["relation"]["text"] not in dict_value["relations"]["texts"]:
                                dict_value["relations"]["texts"].append(triple["relation"]["text"])
                                dict_value["relations"]["ner_lists"].append(triple["relation"]["ner"])
                            if triple["object"]["text"] not in dict_value["objects"]["texts"]:
                                dict_value["objects"]["texts"].append(triple["object"]["text"])
                                dict_value["objects"]["ner_lists"].append(triple["object"]["ner"])
                    logging.info(f"Finished extracting triples information from file {avro_filename} "
                          f"(runtime: {(time.time() - start_timestamp):.2f}s)")
                    reader.close()
            # do manually to save RAM instead of creating entire list object that supports indexing
            values_slice = []
            for i, value in enumerate(sentence_dict.values()):
                values_slice.append(value)
                if len(values_slice) % config["write_every"] == 0 or i == len(sentence_dict) - 1:
                    _write_matches_to_avro(
                        config["output_sentences"]["dir"],
                        config["output_sentences"]["schema_filename"],
                        f"matched_sentences_{file_number:03d}.avro",
                        values_slice
                    )
                    values_slice = []
                    file_number += 1
            del sentence_dict
            sentences = []
            if not tmp_sentence:
                break
        sentences.append(tmp_sentence)
    tmp_file.close()
    os.remove(tmp_filename)


def _extract_sentences_from_opiec():
    config = _read_yaml()
    # enable logging
    log_filename = f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}_sentences_from_opiec.log"
    logging.basicConfig(
        filename=os.path.join(kge_base_dir(), "opiec", "logs", log_filename),
        filemode="w",
        level=logging.DEBUG
    )
    train, valid_test = _read_olpbench_triples(config)
    train_valid_sentences = _find_matches_in_opiec(config, train, valid_test)
    _check_train_valid_sentences(config, train_valid_sentences)
    del train
    del valid_test
    del train_valid_sentences
    _restructure_sentences(config)
    pass


# configure in sentences_from_opiec.yaml
if __name__ == '__main__':
    _extract_sentences_from_opiec()

