import logging
import os
import random
import time
from typing import Tuple, Union, Dict

import yaml
from avro.datafile import DataFileReader
from avro.io import DatumReader
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument, TaggedLineDocument, Doc2Vec

from kge.misc import kge_base_dir
from kge.util.data_stream import WordStream


def _read_yaml() -> dict:
    """
    Read config fine_tune_doc2vec.yaml.
    """
    with open(os.path.join(kge_base_dir(), "kge", "util", "fine_tune_doc2vec.yaml")) as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)


def _extract_entities_and_relations(
        config: dict
) -> Tuple[Union[Dict[str, list], list], Union[Dict[str, list], list]]:
    """
    Read matched triples and create sets for all existing subjects, relations and objects to draw negative samples from.
    If activated, these sets are additionally grouped by the NER tags of the triple parts.
    """
    sample_within_ner = config["negative_sampling"]["sample_within_ner"]
    if sample_within_ner:
        entities = dict()
        relations = dict()
        for ner_tag in config["negative_sampling"]["ner_tags"]:
            entities[ner_tag] = set()
            relations[ner_tag] = set()
    else:
        entities = set()
        relations = set()
    folder = os.path.join(kge_base_dir(), "opiec")
    AVRO_SCHEMA_FILE = os.path.join(folder, config["matched_triples"]["schema_filename"])
    base_dir = config["matched_triples"]["dir"]
    files_in_folder = sorted(os.listdir(os.path.join(folder, base_dir)))
    print(f"Found {len(files_in_folder)} files in folder {base_dir}.")
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, base_dir, avro_filename)
            print(f"Start extracting entities and relations from file {avro_filename}.")
            start_timestamp = time.time()
            reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
            for triple in reader:
                if sample_within_ner:
                    subject_text = triple["subject"]["text"]
                    for subject_ner in triple["subject"]["ner"]:
                        entities[subject_ner].add(subject_text)
                    relation_text = triple["relation"]["text"]
                    for relation_ner in triple["relation"]["ner"]:
                        relations[relation_ner].add(relation_text)
                    object_text = triple["object"]["text"]
                    for object_ner in triple["object"]["ner"]:
                        entities[object_ner].add(object_text)
                else:
                    entities.add(triple["subject"]["text"])
                    relations.add(triple["relation"]["text"])
                    entities.add(triple["object"]["text"])
            print(f"Finished extracting entities and relations from {avro_filename} "
                  f"(runtime: {(time.time() - start_timestamp):.2f}s).")
            reader.close()
    for k, v in entities.items():
        entities[k] = list(v)
    for k, v in relations.items():
        relations[k] = list(v)
    return entities, relations


def _create_negative_samples(
        config: dict,
        entities: Union[dict, list],
        relations: Union[dict, list]
):
    """
    Read matched sentences and create negative samples as specified in the config. The original sentence and negative
    samples are immediately written to disk to minimize active memory usage. They are written to the specified text
    file in format (sentence|X) whereas X is 1 for the original sentence and 0 for the negative samples. From this
    structure, Tagged Documents can be constructed by data_stream.py
    """
    folder = os.path.join(kge_base_dir(), "opiec")
    sentence_dir = config["matched_sentences"]["dir"]
    AVRO_SCHEMA_FILE = os.path.join(folder, config["matched_sentences"]["schema_filename"])
    files_in_folder = sorted(os.listdir(os.path.join(folder, sentence_dir)))
    print(f"Found {len(files_in_folder)} files in folder {sentence_dir}.")
    num_s = config["negative_sampling"]["num_samples"]["s"]
    num_r = config["negative_sampling"]["num_samples"]["r"]
    num_o = config["negative_sampling"]["num_samples"]["o"]
    sample_within_ner = config["negative_sampling"]["sample_within_ner"]
    output_writer = open(os.path.join(folder, config["negative_sampling"]["file"]["filename"]), "w")
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, sentence_dir, avro_filename)
            start_timestamp = time.time()
            counter = 0
            reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
            for triple in reader:
                counter += 1
                output_writer.write(f"{triple['sentence']}|1\n")
                if sample_within_ner:
                    # negative samples for subjects
                    for i in range(num_s):
                        sample = triple['sentence']
                        for subject, ner_tags in zip(triple["subjects"]["texts"], triple["subjects"]["ner_lists"]):
                            sample = sample.replace(subject, random.choice(entities[random.choice(ner_tags)]))
                        output_writer.write(f"{sample}|0\n")
                    # negative samples for relations
                    for i in range(num_r):
                        sample = triple['sentence']
                        for relation, ner_tags in zip(triple["relations"]["texts"], triple["relations"]["ner_lists"]):
                            sample = sample.replace(relation, random.choice(relations[random.choice(ner_tags)]))
                        output_writer.write(f"{sample}|0\n")
                    # negative samples for objects
                    for i in range(num_o):
                        sample = triple['sentence']
                        for obj, ner_tags in zip(triple["objects"]["texts"], triple["objects"]["ner_lists"]):
                            sample = sample.replace(obj, random.choice(entities[random.choice(ner_tags)]))
                        output_writer.write(f"{sample}|0\n")
                else:
                    # negative samples for subjects
                    for i in range(num_s):
                        sample = triple['sentence']
                        for subject in triple["subjects"]["texts"]:
                            sample = sample.replace(subject, random.choice(entities))
                        output_writer.write(f"{sample}|0\n")
                    # negative samples for relations
                    for i in range(num_r):
                        sample = triple['sentence']
                        for relation in triple["relations"]["texts"]:
                            sample = sample.replace(relation, random.choice(relations))
                        output_writer.write(f"{sample}|0\n")
                    # negative samples for objects
                    for i in range(num_o):
                        sample = triple['sentence']
                        for obj in triple["objects"]["texts"]:
                            sample = sample.replace(obj, random.choice(entities))
                        output_writer.write(f"{sample}|0\n")
            reader.close()
            print(f"{counter * (num_s + num_r + num_o)} negative samples created from {avro_filename}"
                  f"(runtime: {(time.time() - start_timestamp):.2f}s).")
    output_writer.close()


def _fine_tune(
        config: dict
):
    """
    Fine tune doc2vec by creating a new doc2vec model from the vocabulary in the matched sentences and negative samples
    and initiating its vectors with a pretrained word2vec model.
    Saves the fine tuned model as the actual model and as its generated embeddings in word2vec format.
    """
    pretrained_dir = os.path.join(kge_base_dir(), "pretrained")
    corpus_file = os.path.join(kge_base_dir(), "opiec", config["negative_sampling"]["file"]["filename"])
    # initiate base doc2vec model
    model = Doc2Vec(
        dm=0,
        dbow_words=1,
        dm_concat=1,
        sample=1e-5,
        hs=0,
        vector_size=config["doc2vec_parameters"]["size"],
        min_count=config["doc2vec_parameters"]["min_count"],
        window=config["doc2vec_parameters"]["window"],
        epochs=config["doc2vec_parameters"]["iter"],
        negative=config["doc2vec_parameters"]["negative"]
    )
    # build corpus from matched sentences
    print("Building vocab from matched sentences...")
    start_time = time.time()
    model.build_vocab(WordStream(corpus_file, shuffle=True))
    print(f"Finished building vocab (required time: {(time.time() - start_time):.2f}s)")
    # manually copy pre-trained embeddings as doc2vec does not support this
    print("Integrating pretrained word2vec embeddings...")
    start_time = time.time()
    pre_trained = KeyedVectors.load_word2vec_format(
        os.path.join(
            pretrained_dir, f"{config['pretrained']['filename']}.{config['pretrained']['filetype']}"
        ),
        binary=False
    )
    for (i, token) in zip(range(len(model.wv.index2entity)), model.wv.index2entity):
        try:
            model.wv.vectors[i] = pre_trained[token]
        except KeyError:
            continue
    del pre_trained
    print(f"Finished integrating pretrained embeddings (required time: {(time.time() - start_time):.2f}s)")
    # fine tune with the matched sentences
    print("Starting fine tuning...")
    start_time = time.time()
    model.train(
        WordStream(corpus_file, shuffle=True),
        total_examples=model.corpus_count,
        epochs=model.epochs
    )
    print(f"Finished fine tuning (required time: {(time.time() - start_time):.2f}s)")
    # save the model
    model.save(os.path.join(pretrained_dir, f"{config['pretrained']['filename']}_doc2vec_fine_tuned.model"))
    model.wv.save_word2vec_format(os.path.join(pretrained_dir, f"{config['pretrained']['filename']}"
                                                               f"_doc2vec_fine_tuned.txt"))


def _fine_tune_doc2vec():
    config = _read_yaml()
    # enable gensim logging
    logging.basicConfig(
        filename=os.path.join(kge_base_dir(), "kge", "util", "fine_tune_log.log"),
        filemode="w",
        level=logging.DEBUG
    )
    if not config["negative_sampling"]["file"]["load_from_file"]:
        entities, relations = _extract_entities_and_relations(config)
        _create_negative_samples(config, entities, relations)
    _fine_tune(config)


# configure in fine_tune_doc2vec.yaml
if __name__ == '__main__':
    _fine_tune_doc2vec()
