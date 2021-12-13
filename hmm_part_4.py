from typing import List, Dict, Union, Set, Tuple
from collections import defaultdict
from main import read_file, generate_emission_matrix, lang, save_prediction, normalise_pair_counts
from itertools import chain
from hmm_part_2 import safe_log


def generate_transition_triplet_matrix(tag_sequences):
    """Generates a transition triplet matrix"""
    tag_triplets = chain.from_iterable(zip(chain(["START", "START"], tag_sequence),
                                           chain(["START"], tag_sequence),
                                           chain(tag_sequence, ["STOP"]))
                                       for tag_sequence in tag_sequences)

    transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    for first, second, third in tag_triplets:
        transition_matrix[first][second][third] += 1
    for v in transition_matrix.values():
        normalise_pair_counts(v)
    return transition_matrix


def find_best(
        word: Union[str, None],
        new_tag: str,
        previous_scores: Dict[str, Dict[str, Tuple[float, List[str]]]],
        e_matrix: Dict[str, Dict[str, float]],
        t_matrix: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Tuple[float, List[str]]]:
    # returns dictionary from prev state to new best score
    output = {}
    for old_tag, old_tag_scores in previous_scores.items():
        max_score = float("-inf")
        max_seq = None
        for old_old_tag, (prev_best, prev_seq) in old_tag_scores.items():
            if word is None:
                e_score = 1
            else:
                e_score = e_matrix[new_tag][word]
            new_score = prev_best + safe_log(t_matrix[old_old_tag][old_tag][new_tag]) + safe_log(e_score)
            if max_score <= new_score:
                max_score = new_score
                max_seq = prev_seq + [new_tag]
        output[old_tag] = (max_score, max_seq)
    return output


def viterbi(
        sentence: List[str],
        e_matrix: Dict[str, Dict[str, float]],
        t_matrix: Dict[str, Dict[str, Dict[str, float]]],
        training_word_set: Set[str],
        n: int = 1
) -> List[str]:
    best_score: List[Dict[str, Dict[str, Tuple[float, List[str]]]]]
    best_score = [{"START": {"START": (0, [])}}]
    for idx, word in enumerate(sentence):
        if word not in training_word_set:
            word = "#UNK#"
        inter_score = {}
        for new_tag in e_matrix.keys():
            new_best = find_best(word, new_tag, best_score[idx], e_matrix, t_matrix)
            inter_score[new_tag] = new_best
        best_score.append(inter_score)
    best_seqs = find_best(None, "STOP", best_score[-1], e_matrix, t_matrix).values()
    max_score = float("-inf")
    max_seq = None
    for score, seq in best_seqs:
        if score >= max_score:
            max_score = score
            max_seq = seq
    return max_seq


if __name__ == '__main__':
    for language in lang:
        train_words, tags, test_words = read_file(language)
        t_matrix = generate_transition_triplet_matrix(tags)
        e_matrix = generate_emission_matrix(train_words, tags, k=1)
        training_word_set = set(chain.from_iterable(train_words))
        all_tags = set(e_matrix.keys())
        predictions = []
        for sentence in test_words:
            prediction = viterbi(sentence, e_matrix, t_matrix, training_word_set)
            predictions.append(prediction)
        save_prediction(test_words, predictions, language, part=4)
