import os
import sys
import time

import pandas as pd

from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
from avro.schema import parse

from kge.misc import kge_base_dir


def _read_olpbench_triples(
    directory: str,
    train_triples_filename: str
) -> set:
    print("Start reading OLPBench triples.")
    data = pd.read_csv(
        os.path.join(directory, train_triples_filename), sep="\t", header=None, usecols=range(0, 3)
    )
    triples = set()
    triples.update(list(data.itertuples(index=False, name=None)))
    print("Finished reading OLPBench triples.")
    return triples


def _find_matches_in_opiec(
        base_dir: str,
        schema_filename: str,
        triples: set
) -> list:
    one_letter_words = {"a", "A", "i", "I"}
    matched_sentences = list()
    folder = os.path.join(kge_base_dir(), "opiec")
    AVRO_SCHEMA_FILE = os.path.join(folder, schema_filename)
    files_in_folder = sorted(os.listdir(os.path.join(folder, base_dir)))
    print(f"Found {len(files_in_folder)} files in folder {base_dir}.")
    for avro_filename in files_in_folder:
        if avro_filename.endswith(".avro"):
            AVRO_FILE = os.path.join(folder, base_dir, avro_filename)
            print(f"Start looking for matches in file {avro_filename}.")
            start_timestamp = time.time()
            reader = DataFileReader(open(AVRO_FILE, "rb"), DatumReader())
            for triple in reader:
                # obtain subject, relation and object, adopted from Samuel Broscheit OLPBench creation code
                # https://github.com/samuelbroscheit/open_knowledge_graph_embeddings/blob/1ce37a4261a37e357a0f4dac3ee130ff11cbea4e/preprocessing/process_avro.py#L16
                # seemingly most efficient version from various timed options
                subject_lc = " ".join([w['word'] if 'QUANT' not in w['word']
                                       else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities']
                                       else w['word'] for w in
                                       sorted(triple['subject'] + triple['dropped_words_subject'], key=lambda x: x['index'])]
                                      ).lower()
                relation_lc = " ".join([w['word'] if 'QUANT' not in w['word']
                                        else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities']
                                        else w['word'] for w in
                                        sorted(triple['relation'] + triple['dropped_words_relation'], key=lambda x: x['index'])]
                                       ).lower()
                object_lc = " ".join([w['word'] if 'QUANT' not in w['word']
                                      else triple['quantities'][w['word'][6:]] if w['word'][6:] in triple['quantities']
                                      else w['word'] for w in
                                      sorted(triple['object'] + triple['dropped_words_object'], key=lambda x: x['index'])]
                                     ).lower()
                # obtain sentence
                sentence = " ".join(
                    [token["word"] for token in triple["sentence_linked"]["tokens"]
                     if len(token["word"]) > 1 or token["word"] in one_letter_words]
                ).lower()

                if (subject_lc, relation_lc, object_lc) in triples:
                    matched_sentences.append({
                        "subject": subject_lc,
                        "relation": relation_lc,
                        "object": object_lc,
                        "sentence": sentence
                    })
            print(f"Found {len(matched_sentences)} matches in {avro_filename} "
                  f"(runtime: {(time.time() - start_timestamp):.2f}s).")
            reader.close()
    return matched_sentences


def _write_matches_to_avro(
        target_dir: str,
        matched_sentences: list,
        schema_filename: str
):
    print("Start writing to avro")
    folder = os.path.join(kge_base_dir(), "opiec")
    AVRO_SCHEMA_FILE = os.path.join(folder, schema_filename)
    schema = parse(open(AVRO_SCHEMA_FILE, "rb").read())
    writer = None
    for i, matched_sentence in enumerate(matched_sentences):
        # create a new file for each 250.000 sentences
        if i % 250000 == 0:
            if writer is not None:
                writer.close()
            filename = f"olpbench_sentences_{(int(i/250000)):03d}.avro"
            print(f"Start writing to file {filename}")
            AVRO_FILE = os.path.join(folder, target_dir, filename)
            writer = DataFileWriter(open(AVRO_FILE, "wb"), DatumWriter(), schema)
        writer.append(matched_sentence)
    print(f"{len(matched_sentences)} matched sentences successfully written to avro files.")


def _extract_sentences_from_opiec(
        dataset: str,
        opiec_files: str
):
    triples = _read_olpbench_triples(
        os.path.join(kge_base_dir(), "data", dataset),
        "train_data_thorough.txt"
    )
    matched_sentences = _find_matches_in_opiec(
        opiec_files,
        "TripleLinked.avsc",
        triples
    )
    _write_matches_to_avro(
        "matched_sentences",
        matched_sentences,
        "MatchedSentences.avsc"
    )
    pass


# give dataset name / folder as first command line argument
# information about correct pickles is hard-coded
if __name__ == '__main__':
    _extract_sentences_from_opiec(sys.argv[1], sys.argv[2])
