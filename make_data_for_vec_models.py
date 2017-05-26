import pandas as pd
from tqdm import tqdm
from string import punctuation
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


model = gensim.models.KeyedVectors.load_word2vec_format('./ruwikiruscorpora_0_300_20.bin', binary=True)

tokens = pd.read_csv('tokens_with_gram_and_dist_v2.csv', encoding='cp1251')
tokens = tokens.drop('Unnamed: 0', 1)

tokens['vec'] = ['']*len(tokens)
pos_tags = {'V':'_VERB','A':'_ADJ','N':'_NOUN','R':'_ADV','Q':'_PART','P':'_PRON','I':'_INTJ',
            'C':'_CCONJ', 'M':'ADV', 'S':'', ',':'', '-':''}
# print(tokens.head())
tokens['vec'] = tokens['vec'].astype('object')
for token_id, token in tqdm(tokens.iterrows()):
    tag = token['lemma']+pos_tags[token['PoS']]
    if tag in model.vocab.keys():
        tokens.set_value(token_id, 'vec', model[tag])


def mean_vec(vectors, count):
    res = []
    for c in range(count):
        if vectors[str(c)] != ['']:
            res.append(vectors[str(c)])
    return np.matrix(res).mean(0).tolist()


anaphora = pd.DataFrame({"target": [], "t_gend": [], "t_count": [], 'dif_cand_disc': [],
                         "c_gend": [], "c_count": [], 'c_pow': [], 'dif_disc_both':[],'dif_plus5':[], 'dif_minus5':[],
                         'dist': [], 'answ': [], 'is_punct': [], 'same_count': [], 'same_gend': []})

not_found = []


def add_anaphora(i, row, anaphora):
    #print(i, row)
    anaphora_local = pd.DataFrame({"target": [], "t_gend": [], "t_count": [], 'dif_cand_disc': [],
                                   "c_gend": [], "c_count": [], 'c_pow': [],'dif_plus5':[], 'dif_minus5':[], 'dif_disc_both':[],
                                   'dist': [], 'answ': [], 'is_punct': [], 'same_count': [], 'same_gend': []})

    vec_count = 0;
    vec_count_n = 0;
    vec_before = {}
    while vec_count_n < 20:
        if tokens.iloc[i - vec_count - 1]['lemma'] not in punctuation:
            if not isinstance(tokens.iloc[i - vec_count - 1]['vec'], str):
                vec_before[str(vec_count_n)] = tokens.iloc[i - vec_count - 1]['vec']
            else:
                vec_before[str(vec_count_n)] = ['']
            vec_count_n += 1
        vec_count += 1
    discource = mean_vec(vec_before, 20)

    vec_count = 0;
    vec_after = {};
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

    t_vec_minus5 = mean_vec(vec_before, 5)
    t_vec_plus5 = mean_vec(vec_after, 5)

    counter_words = 0;
    found_answ = False
    shift = row['shift'];
    target_link = row['link']
    target = row['lemma']

    target_gend = row['gend'];
    target_count = row['count']

    number_added = 0;
    is_punct = 0;
    c = 0
    while counter_words < 25:
        c += 1
        current_posit = i - c - 1
        if tokens.iloc[i - c - 1]['lemma'] in punctuation:
            is_punct = 1
        else:
            candidate_vec = tokens.iloc[i - c - 1]['vec']
            counter_words += 1
            cand_gend = tokens.iloc[i - c - 1]['gend']
            cand_count = tokens.iloc[i - c - 1]['count']
            cand_pos = tokens.iloc[i - c - 1]['PoS']

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
            while vec_count_n < 20:
                if tokens.iloc[current_posit - vec_count - 1]['lemma'] not in punctuation:
                    if not isinstance(tokens.iloc[current_posit - vec_count - 1]['vec'], str):
                        vec_before[str(vec_count_n)] = tokens.iloc[current_posit - vec_count - 1]['vec']
                    else:
                        vec_before[str(vec_count_n)] = ['']
                    vec_count_n += 1
                vec_count += 1

            c_vec_minus5 = mean_vec(vec_before, 5)
            c_vec_plus5 = mean_vec(vec_after, 5)
            discource_cand = mean_vec(vec_before, 20)

            dist = counter_words
            number_added += 1
            if tokens.iloc[i - c - 1]['group_id'] == target_link:
                answ = 1;
                found_answ = True
            else:
                answ = 0
            same_count = (target_count == cand_count)
            same_gend = (target_gend == cand_gend)
            try:
                dif_cand_disc = cosine_similarity(discource, candidate_vec)[0][0]
            except ValueError:
                dif_cand_disc = 0
            try:
                dif_disc = cosine_similarity(discource, discource_cand)[0][0]
            except ValueError:
                dif_disc = 0

            try:
                dif_vec_plus5 = cosine_similarity(c_vec_plus5, t_vec_plus5)[0][0]
            except ValueError:
                dif_vec_plus5 = 0
            # print(len(c_vec_minus3), len(t_vec_minus3), 'minus3')

            try:
                dif_vec_minus5 = cosine_similarity(c_vec_minus5, t_vec_minus5)[0][0]
            except ValueError:
                dif_vec_minus5 = 0
            anaphora_local.loc[len(anaphora_local) + 1] = [answ, cand_count, cand_gend, cand_pos, dif_cand_disc, dif_disc,
                                                           dif_vec_minus5, dif_vec_plus5, dist, is_punct, same_count,
                                                           same_gend, target_count, target_gend, target]
    if found_answ:
        #print('yay')
        #print(pd.concat([anaphora, anaphora_local]))
        return pd.concat([anaphora, anaphora_local])

    else:
        not_found.append([target, shift, i])
        return anaphora

for i, row in tqdm(tokens.loc[tokens['anaph'] == '1.0'].iterrows()):
    i = int(i)
    #print(i, row)
    if row['link'] != '0.0' and row['dist_to_ant'] > 0:
        anaphora = add_anaphora(i, row, anaphora)

    #if i > 100:
    #    break
print(len(not_found))
anaphora = anaphora.fillna('-').reset_index(drop=True)
anaphora.to_csv('data_for_vec_mean_5.csv')