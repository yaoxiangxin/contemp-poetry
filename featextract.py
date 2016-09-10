import collections
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn


porter = nltk.stem.porter.PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')
wnl = nltk.stem.WordNetLemmatizer()
POS_MAPPER = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}


def misc_counts(token_pos_pairs):
    counter = collections.defaultdict(int)
    for token, pos in token_pos_pairs:
        counter[porter.stem(token.lower())] += 1
        counter[pos] += 1
    return [counter['NOUN'], counter['VERB'], counter['ADJ'], counter['ADV'],
            counter['PRON'], counter['i'] + counter['i\'m'], counter['love']]


def senti_scores(token_pos_pairs):
    token_pos_pairs = [
        (wnl.lemmatize(token.lower()), POS_MAPPER[pos])
            for token, pos in token_pos_pairs
                if token not in stopwords
                    and pos in ['VERB', 'NOUN', 'ADJ', 'ADV']
    ]
    pos_scores, neg_scores = [], []
    for token, pos in token_pos_pairs:
        try:
            ss = next(swn.senti_synsets(token, pos)) # most probable synset
        except StopIteration:
            continue
        pos_score, neg_score = ss.pos_score(), ss.neg_score()
        if pos_score:
            pos_scores.append(pos_score)
        if neg_score:
            neg_scores.append(neg_score)
    all_scores = pos_scores + [-neg_score for neg_score in neg_scores]
    return [np.var(all_scores), np.mean(all_scores),
            np.var(pos_scores), np.mean(pos_scores),
            np.var(neg_scores), np.mean(neg_scores)]


if __name__ == '__main__':
    data = []
    for CATEGORY in ['Poet', 'Amateur']:
        FILE = 'Contemp' + CATEGORY + '.txt'
        with open(FILE, 'r') as fin:
            poems = fin.read().split('******')
        del poems[-1]
        for poem in poems:
            raw = poem[poem.find(CATEGORY) + len(CATEGORY):]
            tokens = nltk.tokenize.word_tokenize(raw)
            token_pos_pairs = nltk.tag.pos_tag(tokens, tagset='universal')
            data.append(
                misc_counts(token_pos_pairs.copy()) +
                    senti_scores(token_pos_pairs)
            ) # copy() is redundant but absolutely necessary if senti_scores goes first
    df = pd.DataFrame(
        data,
        columns=['nounFreq', 'verbFreq', 'adjFreq', 'advFreq', 'pronFreq',
                 'i', 'love', 'all_var', 'all_avg', 'pos_var', 'pos_avg',
                 'neg_var', 'neg_avg']
    )
    df.fillna(0)
    df.to_csv('misc-swn.csv', index=False)
