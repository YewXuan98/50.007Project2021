from main import *

def get_prediction(test_words_list, emission_matrix, training_word_set):
    output = []
    for test_list in test_words_list:
        sentence_output = []
        for word in test_list:
            if word in training_word_set:
                tag_assigned = get_best_tag(word, emission_matrix)
            else:
                tag_assigned = get_best_tag("#UNK#", emission_matrix)
            sentence_output.append(tag_assigned)
        output.append(sentence_output)

    return output

if __name__ == '__main__':
    for language in lang:

        train_words, tags, test_words = read_file(language)
        emission_matrix = generate_emission_matrix(train_words, tags, k=1)
        training_word_set = set(chain.from_iterable(train_words))
        prediction = get_prediction(test_words, emission_matrix, training_word_set)
        save_prediction(test_words, prediction, language, part=1)