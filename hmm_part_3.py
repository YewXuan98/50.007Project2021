from itertools import chain
from main import read_file, generate_emission_matrix, generate_transition_matrix, lang, save_prediction
from hmm_part_2 import viterbi
from itertools import chain

if __name__ == '__main__':
    for language in lang:
        train_words, tags, test_words, filein = read_file(language)
        t_matrix = generate_transition_matrix(tags)
        e_matrix = generate_emission_matrix(train_words, tags, k=1)
        training_word_set = set(chain.from_iterable(train_words))
        all_tags = set(e_matrix.keys())
        predictions = []
        for sentence in test_words:
            prediction = viterbi(sentence, e_matrix, t_matrix, training_word_set, n=5)
            predictions.append(prediction)
        save_prediction(test_words, predictions, language, 3, filein)
