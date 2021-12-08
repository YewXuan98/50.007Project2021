from main import  *
from math import log
import sys
from time import sleep
from decimal import Decimal

def multiply(test_word, prev, trans, emi):
    max = sys.float_info.min
    path = ''
    new_prev = {}
    if test_word is None:
        for t0, p in prev.items():
            for t1, t in trans.items():
                if t1 == 'STOP':
                    print(trans.items())
                    new_prev[f'{t0}->{t1}'] = Decimal(p) * Decimal(t)
                    if Decimal(p) * Decimal(t) > max:
                        max = Decimal(p) * Decimal(t)
                        path = f'{t0}->stop'
        new_prev = {}
        new_prev[path] = max
    else:
        if len(prev.items()) == 0:
            for t1, t in trans.items():
                for t2, e in emi.items():
                    if test_word == t2:
                        new_prev[f'start->{t1}'] = Decimal(t) * Decimal(e)
                        if Decimal(t) * Decimal(e) > max:
                            max = Decimal(t) * Decimal(e)
                            path = f'start->{t1}'
            new_prev = {}
            new_prev[path] = max
        else:
            for t0, p in prev.items():
                for t1, t in trans.items():
                    for t2, e in emi.items():
                        if test_word == t2:
                            new_prev[f'{t0}->{t1}'] = Decimal(p) * Decimal(t) * Decimal(e)
                            if Decimal(p) * Decimal(t) * Decimal(e) > max:
                                max = Decimal(p) * Decimal(t) * Decimal(e)
                                path = f'{t0}->{t1}'
            new_prev = {}
            new_prev[path] = max
    # print({tag: {'max': max, 'path': path}})
    print(new_prev)
    sleep(0.5)
    return new_prev

def viterbi(test_words, emission_matrix, tranmission_matrix):
    prev = {}
    min_len = min(len(test_words), len(tranmission_matrix.keys()), len(emission_matrix.keys()))
    print(len(test_words), len(tranmission_matrix.keys()), len(emission_matrix.keys()))
    for idx, transmission in enumerate(tranmission_matrix.items()):
        # end portion (last case->STOPPING PT)
        if idx < min_len:
            if idx == len(tranmission_matrix.keys()) - 1:
                if test_words[idx] not in list(emission_matrix.items())[1][1].keys():
                    test_word = '#UNK#'
                else:
                    test_word = test_words[idx]
                prev = multiply(test_word, prev, list(transmission)[1], None)
                return prev
            if test_words[idx] not in list(emission_matrix.items())[idx][1].keys():
                test_word = '#UNK#'
            else:
                test_word = test_words[idx]
            emission = list(emission_matrix.items())[idx]
            prev = multiply(test_word, prev, list(transmission)[1], list(emission)[1])
        else:
            prev = multiply(None, prev, list(transmission)[1], None)
            return prev

if __name__ == '__main__':
    train_words, tags, test_words = read_file(lang[0])
    t_matrix = generate_transition_matrix(tags)
    e_matrix = generate_emission_matrix(train_words, tags, k=1)
    all_tags = set(e_matrix.keys())
    for word in test_words:
        viterbi(word, e_matrix, t_matrix)
    # print(all_tags)



