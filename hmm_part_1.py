from main import *

if __name__ == '__main__':
    train_words, tags, test_words = read_file(lang[0])

    emission_matrix = generate_emission_matrix(train_words, tags, k=1)

    training_word_set = set(chain.from_iterable(train_words))
    prediction = get_prediction_p1(test_words, emission_matrix, training_word_set)

    save_prediction(lang[0], prediction, part=1)