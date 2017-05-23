import pandas as pd
import gensim
from tqdm import tqdm
import numpy as np

tokens = pd.read_csv('tokens_with_coref.csv')

tokens['hd_shifts'] = tokens['hd_shifts'].fillna(0)
tokens = tokens.drop('anaph', 1).drop('Unnamed: 0', 1).rename(index=str, columns={"hd_shifts": "anaph"}).fillna('-')

model = gensim.models.KeyedVectors.load_word2vec_format('./ruwikiruscorpora_0_300_20.bin', binary=True)
tokens['PoS'] = [g[0] for g in tokens['gram']]
tokens['vec'] = ['']*len(tokens)
pos_tags = {'V': '_VERB', 'A':  '_ADJ','N': '_NOUN', 'R':'_ADV',
            'Q': '_PART', 'P': '_PRON', 'I': '_INTJ', 'C': '_CCONJ',
            'M': 'ADV', 'S': '', ',': '', '-': ''}
# c = 0
# print("STP 1")
tokens['vec'] = tokens['vec'].astype('object')
for token_id, token in tqdm(tokens.iterrows()):
    # c += 1
    tag = token['lemma']+pos_tags[token['PoS']]
    if tag in model.vocab.keys():
        # print(model[tag])
        tokens.set_value(token_id, 'vec', model[tag])
    # if c == 10:
    #    break
# print('STP2')
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity


def mean_vec(vectors, count):
    # print(type(vectors[str(c)]))
    res = []
    for c in range(count):
        if vectors[str(c)] != ['']:
            res.append(vectors[str(c)])
    # res = [vectors[str(c)] for c in range(count)]
    # res.remove([''])
    # print([''] in res)
    # mt = np.matrix(res)
    # print([type(s) for s in res])
    # print(mt.shape)
    # print(1, len(mt.mean(1)), mt.mean(1))
    # print(0, mt.mean(0).shape, mt.mean(0))
    return np.matrix(res).mean(0).tolist()


not_found = []
anaphora = pd.DataFrame({"target": [], "t_gend": [], "t_count": [], 't_vec-3': [], 't_vec-5': [],
                         't_vec+3': [], 't_vec+5': [], 'discourse': [],
                         "cand_vec": [], "c_gend": [], "c_count": [], 'c_pow': [], 'c_vec-3': [], 'c_vec-5': [],
                         'c_vec+3': [], 'c_vec+5': [],
                         'dif_vec-3': [], 'dif_vec-5': [], 'dif_vec+3': [], 'dif_vec+5': [],
                         'dist': [], 'dif_cand_disc': [], 'answ': []})

vec_cls = [ 't_vec-3', 't_vec-5', 't_vec+3', 't_vec+5', 'discourse',
                         "cand_vec", 'c_vec-3', 'c_vec-5', 'c_vec+3', 'c_vec+5',
                         'dif_vec-3', 'dif_vec-5', 'dif_vec+3', 'dif_vec+5',
                         'dif_cand_disc']
for col in vec_cls:
    # print('step7,5')
    anaphora[col] = anaphora[col].astype('object')

# print('POEALI')
def add_anaphora(i, row, anaphora):
    counter_words = 0
    # print(i, row)
    shift = row['shift']
    target_link = row['link']
    target = row['lemma']

    vec_count = 0
    vec_after = {}

    vec_count_n = 0
    while vec_count_n < 5:
        if tokens.iloc[i + vec_count + 1]['lemma'] not in punctuation:
            if len(tokens.iloc[i + vec_count + 1]['vec']) > 10:
                vec_after[str(vec_count_n)] = tokens.iloc[i + vec_count + 1]['vec']
            else:
                vec_after[str(vec_count_n)] = ['']
            # print(vec_count_n, len(vec_after), len(tokens.iloc[i + vec_count + 1]['vec']))
            vec_count_n += 1
        vec_count += 1

    vec_count = 0
    vec_count_n = 0
    vec_before = {}
    while vec_count_n < 30:
        if tokens.iloc[i - vec_count - 1]['lemma'] not in punctuation:
            if not isinstance(tokens.iloc[i - vec_count - 1]['vec'], str):
                vec_before[str(vec_count_n)] = tokens.iloc[i - vec_count - 1]['vec']
            else:
                vec_before[str(vec_count_n)] = ['']

            vec_count_n += 1
        vec_count += 1
    # print(len(vec_after), len(vec_before))
    # print('STEP3')
    discource = mean_vec(vec_before, 30)
    # print(discource)
    # print(['-']*100)
    t_vec_minus3 = mean_vec(vec_before, 3)
    t_vec_minus5 = mean_vec(vec_before, 5)
    t_vec_plus3 = mean_vec(vec_after, 3)
    t_vec_plus5 = mean_vec(vec_after, 5)
    # print(t_vec_minus3)

    if len(row['gram']) > 4:
        target_gend = row['gram'][4]
        target_count = row['gram'][3]
    else:
        target_gend = '-'
        target_count = '-'
    number_added = 0
    found_answ = False
    # print('STEP4')
    while number_added <= 20:
        counter_words += 1
        current_posit = i - counter_words - 1
        if tokens.iloc[current_posit]['lemma'] not in punctuation:
            candidate_vec = tokens.iloc[current_posit]['vec']
            # print(candidate_vec)
            vec_count = 0
            vec_count_n = 0
            vec_after = {}
            while vec_count_n < 5:
                if tokens.iloc[current_posit + vec_count + 1]['lemma'] not in punctuation:
                    if not isinstance(tokens.iloc[current_posit + vec_count + 1]['vec'], str):
                        vec_after[str(vec_count_n)] = tokens.iloc[current_posit + vec_count + 1]['vec']
                    else:
                        vec_after[str(vec_count_n)] = ['']
                    vec_count_n += 1
                vec_count += 1
            vec_count = 0
            vec_before = {}
            vec_count_n = 0
            while vec_count_n < 5:
                if tokens.iloc[current_posit - vec_count - 1]['lemma'] not in punctuation:
                    if not isinstance(tokens.iloc[current_posit - vec_count - 1]['vec'], str):
                        vec_before[str(vec_count_n)] = tokens.iloc[current_posit - vec_count - 1]['vec']
                    else:
                        vec_before[str(vec_count_n)] = ['']
                    vec_count_n += 1
                vec_count += 1
            # print('STEP6')
            c_vec_minus3 = mean_vec(vec_before, 3)
            c_vec_minus5 = mean_vec(vec_before, 5)
            # print(vec_after)
            c_vec_plus3 = mean_vec(vec_after, 3)
            c_vec_plus5 = mean_vec(vec_after, 5)
            # print('STP7')
            # print(c_vec_plus3, t_vec_plus3)#, 'plus3')
            try:
                dif_vec_plus3 = cosine_similarity(c_vec_plus3, t_vec_plus3)
            except ValueError:
                dif_vec_plus3 = 0
            # print(len(c_vec_plus5), len(t_vec_plus5), 'plus5')
            try:
                dif_vec_plus5 = cosine_similarity(c_vec_plus5, t_vec_plus5)
            except ValueError:
                dif_vec_plus5 = 0
            # print(len(c_vec_minus3), len(t_vec_minus3), 'minus3')
            try:
                dif_vec_minus3 = cosine_similarity(c_vec_minus3, t_vec_minus3)
            except ValueError:
                dif_vec_minus3 = 0
            # print(len(c_vec_minus5), len(t_vec_minus5), 'minus5')
            try:
                dif_vec_minus5 = cosine_similarity(c_vec_minus5, t_vec_minus5)
            except ValueError:
                dif_vec_minus5 = 0
            # print(len(discource), len(candidate_vec))
            try:
                dif_cand_disc = cosine_similarity(discource, candidate_vec)
            except ValueError:
                dif_cand_disc = 0

            if len(tokens.iloc[i - counter_words - 1]['gram']) > 4:
                cand_gend = tokens.iloc[i - counter_words - 1]['gram'][4]
                cand_count = tokens.iloc[i - counter_words - 1]['gram'][3]
            else:
                cand_gend = '-'
                cand_count = '-'
            cand_pow = tokens.iloc[i - counter_words - 1]['gram'][0]
            dist = counter_words + 1
            number_added += 1
            # print(number_added)
            if tokens.iloc[i - counter_words - 1]['group_id'] == target_link:
                answ = 1
                found_answ = True
            else:
                answ = 0

            # print('STP8')
            anaphora.loc[len(anaphora) + 1] = [answ, cand_count, cand_gend, cand_pow, c_vec_plus3, c_vec_plus5, c_vec_minus3,
                                               c_vec_minus5, candidate_vec, dif_cand_disc, dif_vec_plus3, dif_vec_plus5,
                                               dif_vec_minus3, dif_vec_minus5, discource, dist, target_count, target_gend,
                                               t_vec_plus3, t_vec_plus5, t_vec_minus3, t_vec_minus5, target]

    if not found_answ:
        # print('WARNING: answ_not_found for ', target, shift, i)
        not_found.append([target, shift, i])
    return anaphora


for i, row in tqdm(tokens.iterrows()):
    counter_words = 0;
    i = int(i)
    if row['anaph'] == 1 and row['link'] != 0:
        anaphora = add_anaphora(i, row, anaphora)
# print(anaphora.head(40))
# print(add_anaphora(68, tokens.iloc[68], anaphora))

anaphora.to_csv('anaphora.csv')
