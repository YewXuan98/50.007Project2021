# import pandas as pd
from itertools import chain
from collections import defaultdict
import argparse

lang = ['ES', 'RU']

def read_file(lang):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="train_in", help="train file")
    parser.add_argument("-i", dest="test_in", help="input file")
    args = parser.parse_args()
    train_in = args.train_in
    test_in = args.test_in
    data_type = [train_in, test_in]
    train_words = []
    tags = []
    test_words = []

    for i in range(2):
        with open(f'{lang}/{data_type[i]}', 'r', encoding="utf8") as fin:
            doc = fin.read().strip()
            blocks = doc.split('\n\n')

            if i == 0:
                for block in blocks:
                    ws = []
                    ts = []
                    for line in block.split('\n'):
                        w, t = line.rsplit(' ', 1)
                        ws.append(w)
                        ts.append(t)
                    train_words.append(ws)
                    tags.append(ts)
            elif i == 1:
                for block in blocks:
                    ws = []
                    for w in block.split('\n'):
                        ws.append(w)
                    test_words.append(ws)

    return train_words, tags, test_words, test_in


def count_y(tag, tag_seq_ls):
    tag_seq_ls_flattened = [chain.from_iterable(tag_seq_ls)]

    return tag_seq_ls_flattened.count(tag)


def count_pairs(pairs):
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    for first, second in pairs:
        counts[first][second] += 1
    return counts


def normalise_pair_counts(matrix):
    for vec in matrix.values():
        count = sum(vec.values())
        for k in vec:
            vec[k] /= count


def generate_emission_matrix(word_seqs, tag_seqs, k):
    """Generates a nested dictionary with the estimated emission parameters.
    output[tag][word] is the emission param for the given tag, word pair."""
    tag_word_pairs = chain.from_iterable(zip(*seq_pair) for seq_pair in zip(tag_seqs, word_seqs))

    emission_matrix = count_pairs(tag_word_pairs)
    for tag in emission_matrix:
        emission_matrix[tag]["#UNK#"] = k

    normalise_pair_counts(emission_matrix)
    return emission_matrix


def generate_transition_matrix(tag_sequences):
    """Generates a nested dictionary with the estimated transition parameters.
    output[tag1][tag2] is the transition param from tag1 to tag2."""
    # get all tag pairs
    tag_pairs = chain.from_iterable(zip(chain(["START"], tag_sequence), chain(tag_sequence, ["STOP"]))
                                    for tag_sequence in tag_sequences)

    transition_matrix = count_pairs(tag_pairs)
    normalise_pair_counts(transition_matrix)

    return transition_matrix


# method to check for the tag corresponding to the best score for the word
def get_best_tag(word, emission_matrix):
    y = ""
    score_max = -1

    for tag, emission_matrix_row in emission_matrix.items():
        # get the score corresponding to the word for B-positive, B-negative...
        score_current = emission_matrix_row[word]

        if score_current > score_max:
            score_max = score_current
            y = tag
    return y


def save_prediction(test_words, predictions, lang, part, test_in):
    if test_in == 'dev.in':
        with open(f"{lang}/dev.p{part}.out", "w") as f:
            for sentence, tags in zip(test_words, predictions):
                for word, tag in zip(sentence, tags):
                    f.write(f"{word} {tag}\n")
                f.write("\n")
    elif test_in == 'test.in':
        with open(f"{lang}/test.p{part}.out", "w") as f:
            for sentence, tags in zip(test_words, predictions):
                for word, tag in zip(sentence, tags):
                    f.write(f"{word} {tag}\n")
                f.write("\n")