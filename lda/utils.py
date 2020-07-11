import gensim
from gensim.models import CoherenceModel
class Utils:
    @staticmethod
    def optimaze_topics_amount(dictionary, corpus, texts, limit, alpha, betta, start=2, step=3):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                                   id2word=dictionary,
                                                   num_topics=3,
                                                   random_state=100,
                                                   chunksize=100,
                                                   passes=10,
                                                   per_word_topics=True)
            model_list.append(lda_model)

            coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary,
                                                 coherence='c_v')
            coherence_values.append(coherence_model_lda.get_coherence())

        return model_list, coherence_values