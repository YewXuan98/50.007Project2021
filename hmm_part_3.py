from hmm_part_2 import *

if __name__ == '__main__':
    train_words, tags, test_words = read_file(lang[0])
    t_matrix = generate_transition_matrix(tags)
    e_matrix = generate_emission_matrix(train_words, tags, k=1)
    training_word_set = set(chain.from_iterable(train_words))
    all_tags = set(e_matrix.keys())
    predictions = []
    for sentence in test_words:
        prediction = viterbi(sentence, e_matrix, t_matrix, training_word_set, n=5)
        predictions.append(prediction)
    save_prediction(test_words, predictions, lang[0], part=3)
    # print(all_tags)
