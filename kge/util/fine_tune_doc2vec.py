import datetime
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
    logging.info(f"Found {len(files_in_folder)} files in folder {base_dir}.")
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, base_dir, avro_filename)
            logging.info(f"Start extracting entities and relations from file {avro_filename}.")
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
            logging.info(f"Finished extracting entities and relations from {avro_filename} "
                  f"(runtime: {(time.time() - start_timestamp):.2f}s).")
            reader.close()
    for k, v in entities.items():
        entities[k] = list(v)
    for k, v in relations.items():
        relations[k] = list(v)
    return entities, relations


def _sample_sentence(
        num_rep: int,
        sentence: str,
        to_replace: dict,
        replace_list: Union[list, dict],
        output_writer
):
    """
    Create negative samples from a given sentence.
    """
    for i in range(num_rep):
        sample = sentence
        for replace_text, ner_tags in zip(to_replace["texts"], to_replace["ner_lists"]):
            if isinstance(replace_list, list):  # do not replace within ner tag
                sample = sample.replace(replace_text, random.choice(replace_list))
            else:  # replace within ner tag
                sample = sample.replace(replace_text, random.choice(replace_list[random.choice(ner_tags)]))
        sample = " ".join([word for word in sample.split() if word.isalpha()])  # clean sentence
        document_text = sample + "_|_0"
        output_writer.write(document_text + "\n")


def _create_negative_samples(
        config: dict,
        entities: Union[dict, list],
        relations: Union[dict, list]
):
    """
    Read matched sentences and create negative samples as specified in the config. The original sentence and negative
    samples are immediately written to disk to minimize active memory usage. They are written to the specified text
    file in format (sentence_|_X) whereas X is 1 for the original sentence and 0 for the negative samples. From this
    structure, Tagged Documents can be constructed by data_stream.py
    """
    folder = os.path.join(kge_base_dir(), "opiec")
    sentence_dir = config["matched_sentences"]["dir"]
    AVRO_SCHEMA_FILE = os.path.join(folder, config["matched_sentences"]["schema_filename"])
    files_in_folder = sorted(os.listdir(os.path.join(folder, sentence_dir)))
    logging.info(f"Found {len(files_in_folder)} files in folder {sentence_dir}.")
    num_s = config["negative_sampling"]["num_samples"]["s"]
    num_r = config["negative_sampling"]["num_samples"]["r"]
    num_o = config["negative_sampling"]["num_samples"]["o"]
    sample_within_ner = config["negative_sampling"]["sample_within_ner"]
    tmp_filepath = os.path.join(folder, f"tmp_{config['negative_sampling']['file']['filename']}")
    output_writer = open(tmp_filepath, "w")
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, sentence_dir, avro_filename)
            start_timestamp = time.time()
            counter = 0
            reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
            for triple in reader:
                counter += 1
                document_text = " ".join([word for word in triple['sentence'].split() if word.isalpha()]) + "_|_1"
                output_writer.write(document_text + "\n")
                _sample_sentence(num_s, triple["sentence"], triple["subjects"], entities, output_writer)
                _sample_sentence(num_r, triple["sentence"], triple["relations"], relations, output_writer)
                _sample_sentence(num_o, triple["sentence"], triple["objects"], entities, output_writer)
            reader.close()
            logging.info(f"{counter * (num_s + num_r + num_o)} negative samples created from {avro_filename}"
                  f"(runtime: {(time.time() - start_timestamp):.2f}s).")
    output_writer.close()
    logging.info("Start shuffling samples...")
    start_timestamp = time.time()
    output_writer = open(os.path.join(folder, config['negative_sampling']['file']['filename']), "w")
    for line in WordStream(tmp_filepath, shuffle=True, as_tagged_doc=False):
        output_writer.write(f"{line}\n")
    output_writer.close()
    os.remove(tmp_filepath)
    logging.info(f"Negative samples shuffled (runtime: {(time.time() - start_timestamp):.2f}s).")


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
        vector_size=config["doc2vec"]["parameters"]["size"],
        min_count=config["doc2vec"]["parameters"]["min_count"],
        window=config["doc2vec"]["parameters"]["window"],
        epochs=config["doc2vec"]["parameters"]["iter"],
        negative=config["doc2vec"]["parameters"]["negative"]
    )
    # build corpus from matched sentences
    logging.info("Building vocab from matched sentences...")
    corpus = WordStream(corpus_file, shuffle=False, as_tagged_doc=True)
    start_time = time.time()
    model.build_vocab(corpus)
    logging.info(f"Finished building vocab (required time: {(time.time() - start_time):.2f}s)")
    # manually copy pre-trained embeddings as doc2vec does not support this
    logging.info("Integrating pretrained word2vec embeddings...")
    start_time = time.time()
    pre_trained = KeyedVectors.load_word2vec_format(
        os.path.join(
            pretrained_dir, f"{config['doc2vec']['pretrained']['filename']}."
                            f"{config['doc2vec']['pretrained']['filetype']}"
        ),
        binary=False
    )
    for (i, token) in zip(range(len(model.wv.index2entity)), model.wv.index2entity):
        try:
            model.wv.vectors[i] = pre_trained[token]
        except KeyError:
            continue
    del pre_trained
    logging.info(f"Finished integrating pretrained embeddings (required time: {(time.time() - start_time):.2f}s)")
    # fine tune with the matched sentences
    logging.info("Starting fine tuning...")
    start_time = time.time()
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )
    logging.info(f"Finished fine tuning (required time: {(time.time() - start_time):.2f}s)")
    # save the model
    model.save(os.path.join(pretrained_dir, f"{config['doc2vec']['pretrained']['filename']}_doc2vec_fine_tuned.model"))
    model.wv.save_word2vec_format(os.path.join(pretrained_dir, f"{config['doc2vec']['pretrained']['filename']}"
                                                               f"_doc2vec_fine_tuned.txt"))


def _fine_tune_doc2vec():
    config = _read_yaml()
    # enable logging
    log_filename = f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}_doc2vec_fine_tune.log"
    logging.basicConfig(
        filename=os.path.join(kge_base_dir(), "opiec", "logs", log_filename),
        filemode="w",
        level=logging.DEBUG
    )
    if not config["negative_sampling"]["file"]["load_from_file"]:
        entities, relations = _extract_entities_and_relations(config)
        _create_negative_samples(config, entities, relations)
    if config["doc2vec"]["fine_tune"]:
        _fine_tune(config)


# configure in fine_tune_doc2vec.yaml
if __name__ == '__main__':
    _fine_tune_doc2vec()
