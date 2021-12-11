from typing import List, Dict, Union, Set, Tuple

from main import *
from math import log


def safe_log(n):
    return float("-inf") if n == 0 else log(n)


def find_best_n(
        word: Union[str, None],
        new_tag: str,
        previous_scores: Dict[str, List[Tuple[float, List[str]]]],
        e_matrix: Dict[str, Dict[str, float]],
        t_matrix: Dict[str, Dict[str, float]],
        n: int = 1
) -> List[Tuple[float, List[str]]]:
    possible_scores = []
    for old_tag in previous_scores.keys():
        for prev_best, prev_seq in previous_scores[old_tag]:
            if word is None:
                e_score = 1
            else:
                e_score = e_matrix[new_tag][word]
            new_score = prev_best + safe_log(t_matrix[old_tag][new_tag]) + safe_log(e_score)
            possible_scores.append((new_score, prev_seq + [new_tag]))
    possible_scores.sort(reverse=True)
    return possible_scores[:n]


def viterbi(
        sentence: List[str],
        e_matrix: Dict[str, Dict[str, float]],
        t_matrix: Dict[str, Dict[str, float]],
        training_word_set: Set[str],
        n: int = 1
) -> List[str]:
    best_score: List[Dict[str, List[Tuple[float, List[str]]]]]
    best_score = [{"START": [(0, [])]}]
    for idx, word in enumerate(sentence):
        if word not in training_word_set:
            word = "#UNK#"
        inter_score = {}
        for new_tag in e_matrix.keys():
            new_best = find_best_n(word, new_tag, best_score[idx], e_matrix, t_matrix, n)
            inter_score[new_tag] = new_best
        best_score.append(inter_score)
    best_n = find_best_n(None, "STOP", best_score[-1], e_matrix, t_matrix, n)
    nth_best_score, nth_best_seq = best_n[-1]
    return nth_best_seq[:-1]


if __name__ == '__main__':
    train_words, tags, test_words = read_file(lang[0])
    t_matrix = generate_transition_matrix(tags)
    e_matrix = generate_emission_matrix(train_words, tags, k=1)
    training_word_set = set(chain.from_iterable(train_words))
    all_tags = set(e_matrix.keys())
    predictions = []
    for sentence in test_words:
        prediction = viterbi(sentence, e_matrix, t_matrix, training_word_set)
        predictions.append(prediction)
    save_prediction(test_words, predictions, lang[0], part=2)
    # print(all_tags)
