import numpy as np
import gensim
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim
# #import pickle
import pyLDAvis
from gensim.models import CoherenceModel
class Model:
    def __init__(self, sample_length, alpha, betta):
        self.sample_length = sample_length
        self.alpha = alpha
        self.betta = betta

    def create_corpus(self, texts_list):
        new_corpus = []
        min_len_text = self.sample_length
        for text in texts_list:
            if len(text) > min_len_text:
                new_corpus.append(text)
        cropped_texts = []
        for text in new_corpus:
            cropped_texts.append(text[:min_len_text])

        text_matrix = list(np.vstack((list(text) for text in cropped_texts)))

        id2word = corpora.Dictionary(text_matrix)
        corpus = [id2word.doc2bow(text) for text in text_matrix]

        return corpus, id2word, text_matrix

    def train(self, text_list):
        corpus, id2word, text_matrix = self.create_corpus(text_list)
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=5,
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               per_word_topics=True)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=text_matrix, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)


        with open('non_processed_model_alpha_'+ str(self.alpha) + '_betta_'+str(self.betta) + '_samplesize_' + str(self.sample_length)+'.txt'  , 'w+', encoding='utf-8', errors='ignore') as out:
            pprint(lda_model.print_topics(), stream=out)
            out.write('\nCoherence Score: ' + str(coherence_lda))
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(LDAvis_prepared, 'non_processed_model_alpha_'+ str(self.alpha) + '_betta_'+str(self.betta) + '_samplesize_' + str(self.sample_length)+'.html')
