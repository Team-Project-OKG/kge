"""
Module for instantiating a data stream with fast, random sampling.

From https://indico.io/blog/fast-method-stream-data-from-big-data-sources/, adjusted and bug fixed
"""

import os
import time
import mmap
import numpy as np
from gensim.models.doc2vec import TaggedDocument


def create_tagged_document(s):
    """
    Create TaggedDocument from line
    """
    words, tag = s.decode("utf-8").split("|")
    return TaggedDocument(words.split(), [int(tag)])


class WordStream(object):
    """
    Stream words from a corpus of newline-delimited text. Single-threaded
    version. Works on input larger than RAM, as long as the number of lines
    doesn't cause an int overflow. No worries if you're using a filesystem
    with 64-bit addressing.
    This version uses numpy for compactness of `uint64` arrays (for offsets).
    If you can't afford the numpy dependency but have memory to spare,
    plain python lists work too.
    Example usage:
        words = [x for x in WordStream('corpus.txt', shuffle = True)]
    """

    def __init__(self,
                 source,
                 offsets=None,
                 shuffle=True,
                 seed=2,
                 log_each=int(5e6)):
        np.random.seed(seed)
        self.source = source  # string defining a path to a file-like object
        self.log_each = int(log_each)  # int defining the logging frequency
        self.filesize = int(os.stat(source).st_size)
        self.shuffle = shuffle
        self.seed = seed
        print("Reading %d bytes of data from source: '%s'" % (self.filesize,
                                                              self.source))
        if offsets:
            print("Using offsets that were given as input")
        else:
            print("No pre-computed offsets detected, scanning file...")
            offsets = self.scan_offsets()  # expect a numpy array to be returned here
        self.offsets = offsets
        if self.shuffle:
            np.random.shuffle(self.offsets)
            print("offsets shuffled using random seed: %d" % self.seed)

    def __iter__(self):
        """
        Yields a list of words for each line in the data source.
        If user wants to pass over the data multiple times (i.e., multiple
        epochs), shuffle the offsets each time, and pass the offsets
        explicitly when re-instantiating the generator.
        To keep concurrent workers busy, rewrite this as a generator that
        yields offsets into a queue (instead of `enumerate`)...then this
        generator can consume offsets from the queue.
        """
        with open(self.source, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            filesize = os.stat(self.source).st_size  # get filesize
            len_offsets = len(self.offsets)  # compute once
            for line_number, offset in enumerate(self.offsets):  # traverse random path
                if line_number % self.log_each == 0:
                    print(f"{line_number} lines yielded")
                if int(line_number) >= len_offsets:
                    print("Error at line number: %d" % line_number)
                    continue
                offset_begin = self.offsets[line_number]
                try:
                    mm.seek(offset_begin)
                    line = mm.readline()
                except:
                    print("Error at location: %d" % offset)
                    continue
                if len(line) == 0:
                    continue  # no point to returning an empty list (i.e., whitespace)
                yield create_tagged_document(line)  # chain parsing logic/functions here

    def scan_offsets(self):
        """
        Scan file to find byte offsets
        """
        tic = time.time()
        tmp_offsets = []  # python auto-extends this
        print("Scanning file '%s' to find byte offsets for each line..." % self.source)
        with open(self.source, "r+b") as f:
            i = 0  # technically, this can grow unbounded...practically, not an issue
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)  # lazy eval-on-demand via POSIX filesystem
            tmp_offsets.append(0)
            for line in iter(mm.readline, ''):
                if len(line) == 0:
                    break
                pos = mm.tell()
                tmp_offsets.append(pos)
                i += 1
                if i % self.log_each == 0:
                    print("%dM examples scanned" % (i / 1e6))
            mm.close()
        toc = time.time()
        offsets = np.asarray(tmp_offsets,
                             dtype='uint64')  # convert to numpy array for compactness; can use uint32 for small and medium corpora (i.e., less than 100M lines)
        del tmp_offsets  # don't need this any longer, save memory
        print("...file has %d bytes and %d lines" % (self.filesize, i))
        print("%.2f seconds elapsed scanning file for offsets" % (toc - tic))
        return offsets
