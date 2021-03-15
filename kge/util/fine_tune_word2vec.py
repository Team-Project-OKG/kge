import datetime
import logging
import os
import random
import time

import yaml
from avro.datafile import DataFileReader
from avro.io import DatumReader
from gensim.models import Word2Vec

from kge.misc import kge_base_dir


def _read_yaml() -> dict:
    """
    Read config fine_tune_word2vec.yaml.
    """
    with open(os.path.join(kge_base_dir(), "kge", "util", "fine_tune_word2vec.yaml")) as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)


def _extract_sentences(
        base_dir: str,
        schema_filename: str
) -> list:
    """
    Get unique list of sentences from matches found in OPIEC. Supports using matches for each triple or each sentence,
    whereas the second option is obviously more efficient.
    """
    sentences = set()
    folder = os.path.join(kge_base_dir(), "opiec")
    AVRO_SCHEMA_FILE = os.path.join(folder, schema_filename)
    files_in_folder = sorted(os.listdir(os.path.join(folder, base_dir)))
    logging.info(f"Found {len(files_in_folder)} files in folder {base_dir}.")
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, base_dir, avro_filename)
            logging.info(f"Start extracting sentences from file {avro_filename}.")
            start_timestamp = time.time()
            reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
            for triple in reader:
                # add to set instead of appending to list so both outputs of sentences_from_opiec can be used
                sentences.add(tuple([word for word in triple["sentence"].split() if word.isalpha()]))
            logging.info(f"Finished extracting sentences from {avro_filename} (runtime: {(time.time() - start_timestamp):.2f}s"
                         f"; total sentences extracted: {len(sentences)}).")
            reader.close()
    logging.info(f"{len(sentences)} sentences extracted.")
    return list(sentences)


def _fine_tune(
        matched_sentences: list,
        pretrained_filename: str,
        pretrained_filetype: str,
        word2vec_parameters: dict
):
    """
    Fine tune word2vec by creating a new word2vec model from the vocabulary in the matched sentences and initiating its
    vectors with a pretrained word2vec model.
    Saves the fine tuned model as the actual model and as its generated embeddings in word2vec format.
    """
    pretrained_dir = os.path.join(kge_base_dir(), "pretrained")
    # initiate base word2vec model
    model = Word2Vec(
        size=word2vec_parameters["size"],
        min_count=word2vec_parameters["min_count"],
        sg=word2vec_parameters["sg"],
        window=word2vec_parameters["window"],
        iter=word2vec_parameters["iter"],
        negative=word2vec_parameters["negative"]
    )
    # shuffle matched sentences to avoid clusters
    logging.info("Shuffling training data...")
    start_time = time.time()
    random.shuffle(matched_sentences)
    logging.info(f"Finished shuffling training data (required time: {(time.time() - start_time):.2f}s)")
    # build corpus from matched sentences
    logging.info("Building vocab from matched sentences...")
    start_time = time.time()
    model.build_vocab(matched_sentences)
    logging.info(f"Finished building vocab (required time: {(time.time() - start_time):.2f}s)")
    # load pretrained word2vec embeddings
    logging.info("Integrating pretrained word2vec embeddings...")
    start_time = time.time()
    model.intersect_word2vec_format(
        os.path.join(pretrained_dir, f"{pretrained_filename}.{pretrained_filetype}"),
        binary=(pretrained_filetype == "bin"),
        lockf=1.0  # allow further training of pretrained word2vec vectors
    )
    logging.info(f"Finished integrating pretrained embeddings (required time: {(time.time() - start_time):.2f}s)")
    # fine tune with the matched sentences
    logging.info("Starting fine tuning...")
    start_time = time.time()
    model.train(
        matched_sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )
    logging.info(f"Finished fine tuning (required time: {(time.time() - start_time):.2f}s)")
    # save the model
    model.save(os.path.join(pretrained_dir, f"{pretrained_filename}_fine_tuned.model"))
    model.wv.save_word2vec_format(os.path.join(pretrained_dir, f"{pretrained_filename}_fine_tuned.txt"))


def _fine_tune_word2vec():
    config = _read_yaml()
    # enable logging
    log_filename = f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}_word2vec_fine_tune.log"
    logging.basicConfig(
        filename=os.path.join(kge_base_dir(), "opiec", "logs", log_filename),
        filemode="w",
        level=logging.DEBUG
    )
    matched_sentences = _extract_sentences(
        config["matched_sentences"]["dir"],
        config["matched_sentences"]["schema_filename"]
    )
    _fine_tune(
        matched_sentences,
        config["word2vec"]["pretrained"]["filename"],
        config["word2vec"]["pretrained"]["filetype"],
        config["word2vec"]["parameters"]
    )


# configure in fine_tune_word2vec.yaml
if __name__ == '__main__':
    _fine_tune_word2vec()
