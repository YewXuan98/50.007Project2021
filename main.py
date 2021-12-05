# import pandas as pd
import math
from itertools import chain
from collections import defaultdict

lang = ['ES','RU']
data_type = ['train', 'dev.in']


def read_file(lang):
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

    return train_words, tags, test_words


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


def generate_emission_matrix(word_seqs, tag_seqs):
    k = 1

    tag_word_pairs = chain.from_iterable(zip(*seq_pair) for seq_pair in zip(tag_seqs, word_seqs))

    emission_matrix = count_pairs(tag_word_pairs)
    for tag in emission_matrix:
        emission_matrix[tag]["#UNK#"] = k

    normalise_pair_counts(emission_matrix)
    return emission_matrix


def generate_transition_matrix(tag_sequences):
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


def get_prediction(test_words_list, emission_matrix, training_word_set):
    output = ""
    for test_list in test_words_list:
        for word in test_list:
            tag_assigned = ""
            if word in training_word_set:
                tag_assigned = get_best_tag(word, emission_matrix)
            else:
                tag_assigned = get_best_tag("#UNK#", emission_matrix)

            output += f"{word} {tag_assigned}"
            output += "\n"
        output += "\n"

    return output


def save_prediction(lang, prediction):
    with open(f"{lang}/dev.p2.out", "w") as f:
        f.write(prediction)


def safe_log(n):
    if n == 0:
        return float("-inf")
    else:
        return math.log(n)


def viterbi_prediction(observations, emission_matrix, transmission_matrix):
    # tag -> (log score, sequence)
    previous_scores = {"START": (0, [])}
    tags = emission_matrix.keys()
    for observation in observations:

        # tag -> (log score, sequence)
        updated_scores = {}

        for tag in tags:
            emission_score = safe_log(emission_matrix[tag][observation])
            max_score = float("-inf")
            max_seq = None

            for previous_tag, (previous_score, previous_seq) in previous_scores.items():
                transition_score = safe_log(transmission_matrix[previous_tag][tag])
                score = emission_score + previous_score + transition_score
                if score > max_score:
                    max_score = score
                    max_seq = previous_seq + [tag]

            if max_seq is not None:
                updated_scores[tag] = (max_score, max_seq)

        previous_scores = updated_scores

    # handle final STOP
    max_score = float("-inf")
    max_seq = None

    for previous_tag, (previous_score, previous_seq) in previous_scores.items():
        transition_score = safe_log(transmission_matrix[previous_tag]["STOP"])
        score = previous_score + transition_score
        if score > max_score:
            max_score = score
            max_seq = previous_seq

    return max_seq


if __name__ == '__main__':
    train_words, tags, test_words = read_file(lang[0])

    k = 1
    emission_matrix = generate_emission_matrix(train_words, tags)

    training_word_set = set(chain.from_iterable(train_words))
    prediction = get_prediction(test_words, emission_matrix, training_word_set)

    save_prediction(lang[0], prediction)
