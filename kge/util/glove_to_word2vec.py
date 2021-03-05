import os
import sys

from gensim.scripts.glove2word2vec import glove2word2vec

from kge.misc import kge_base_dir


def _convert_to_word2vec(
        filename: str
):
    folder = os.path.join(kge_base_dir(), "pretrained")
    input_file = os.path.join(folder, filename)
    index = filename.rindex(".")
    output_file = filename[0:index] + "_word2vec" + filename[index:len(filename)]
    output_file = os.path.join(folder, output_file)
    glove2word2vec(input_file, output_file)


# give file name as first command line argument
if __name__ == '__main__':
    _convert_to_word2vec(sys.argv[1])
