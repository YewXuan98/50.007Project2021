from tqdm import tqdm
import itertools

languages = ["ES", "RU"]

lang = ['ES', 'RU']
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
                        w, t = line.split(' ')
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
# for language in languages:
#
#     tag_seq_ls, word_seq_ls, test_word_seq_ls = load_dataset(language)

tr_words, tags, te_words = read_file(lang[0])

#print(tag_seq_ls_ru)
print(te_words)