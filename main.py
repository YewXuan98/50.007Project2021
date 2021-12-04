# import pandas as pd
from itertools import chain, product

# tr - train
# te - test

lang = ['ES','RU']
data_type = ['train', 'dev.in']


def read_file(lang):
    tr_words = []
    tags = []
    te_words = []

    for i in range(2):
        with open(f'{lang}/{data_type[i]}', 'r') as fin:
            doc = fin.read().strip()
            blocks = doc.split('\n\n')

            if i == 0:
                for block in blocks:
                    ws = []
                    ts = []
                    for line in block.split('\n'):
                        w, t = line.rsplit(' ',1)
                        ws.append(w)
                        ts.append(t)
                    tr_words.append(ws)
                    tags.append(ts)
                fin.close()

            elif i == 1:
                for block in blocks:
                    ws = []
                    for w in block.split('\n'):
                        ws.append(w)
                    te_words.append(ws)
                fin.close()

    return tr_words, tags, te_words


def get_sorted_unique_elements(ls):
    #chain will combine all the many segmented list to combine into one
    u_ls = list(set(chain.from_iterable(ls)))
    u_ls.sort()

    return u_ls


def get_unique_tags(tags):
    u_tags = get_sorted_unique_elements(tags)
    sorted_tags = ["START"] + u_tags + ["STOP"]
    return u_tags, sorted_tags


def get_unique_words(words):
    u_words = get_sorted_unique_elements(words)
    return u_words


def gen_emission_pairs(u_tags, u_words):
    emission_pairs = []

    for tag, word in zip(u_tags, u_words):
        for t, w in zip(tag, word):
            emission_pairs.append([t, w])

    return emission_pairs


def gen_possible_emission_pairs(u_tags, u_words):
    return [product(u_tags, u_words)]


def count_y(tag, tag_seq_ls):
    tag_seq_ls_flattened = [chain.from_iterable(tag_seq_ls)]

    return tag_seq_ls_flattened.count(tag)


def gen_emission_matrix(tags_unique, words_unique, tag_seq_ls, word_seq_ls, k):

    # create and initialise emission matrix
    emission_matrix = {}
    for tag in tags_unique:
        emission_matrix_row = {}
        for word in words_unique:
            emission_matrix_row[word] = 0.0
        emission_matrix_row["#UNK#"] = 0.0
        emission_matrix[tag] = emission_matrix_row

    # population emission matrix with counts
    for tags, words in zip(tag_seq_ls, word_seq_ls):
        for tag, word in zip(tags, words):
            emission_matrix[tag][word] += 1

    # divide cells by sum, to get probability
    for tag, emission_matrix_row in emission_matrix.items():
        row_sum = count_y(tag, tag_seq_ls) + k

        # words in training set
        popped = emission_matrix_row.popitem()
        for word, cell in emission_matrix_row.items():
            emission_matrix[tag][word] = cell / row_sum

        # word == #UNK#
        emission_matrix[tag]["#UNK#"] = k / (row_sum)

    return emission_matrix

#count(y->x)/count(y)
def estimate_emission_param():
    emission_matrix = {}
    #for each unique tag we will initialise the emission prob w
    for t in u_tags:
        emission_matrix_row = {}
        for word in u_words:
            emission_matrix_row[word] = 0.0
        emission_matrix[t] = emission_matrix_row



    for tag, word in zip(tags, tr_words):
        for t, w in zip(tag, word):
            emission_matrix[t][w] += 1


    for t, emission_matrix_row in emission_matrix.items():
        row_sum = sum(emission_matrix_row.values())
    count_y(t, tags)
    for word, cell in emission_matrix_row.items():
        emission_matrix[t][word] = cell / row_sum

    #print(emission_matrix)


#method to check for the tag corresponding to the best score for the word
def get_best_tag(word, emission_matrix):
    y = ""
    score_max = -1

    for tag, emission_matrix_row in emission_matrix.items():
        #get the score corresponding to the word for B-positive, B-negative...
        score_current = emission_matrix_row[word]

        if score_current > score_max:
            score_max = score_current
            y = tag
    return y

def get_prediction(test_words_list, emission_matrix):
    output = ""
    for test_list in test_words_list:
        for word in test_list:
            tag_assigned = ""
            if word in words_not_found_in_test_list:
                tag_assigned = get_best_tag("#UNK#", emission_matrix)
            else:
                tag_assigned = get_best_tag(word, emission_matrix)

            output += f"{word} {tag_assigned}"
            output += "\n"
        output += "\n"

    return output

def save_prediction(lang, prediction):
    with open(f"{lang}/dev.p2.out", "w") as f:
        f.write(prediction)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tr_words, tags, te_words = read_file(lang[0])
    u_tags, u_tag_w_start_stop = get_unique_tags(tags)
    u_words = get_unique_words(tr_words)

    # print(te_words)

    emission_pairs = gen_emission_pairs(tags, tr_words)
    #print(estimate_emission_param())
    emission_pairs_possible = gen_possible_emission_pairs(
        u_tags, u_words)

    k = 1
    emission_matrix = gen_emission_matrix(u_tags, u_words,
                                          tags, tr_words, k)

    #get the unique words in test list
    test_word_unique = get_sorted_unique_elements(te_words)
    #print(test_word_unique)

    words_not_found_in_test_list = set(test_word_unique).difference(set(u_words))
    #print(words_not_found_in_test_list)
    prediction = get_prediction(te_words, emission_matrix)

    #print(prediction)

    save_prediction(lang[0], prediction)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/