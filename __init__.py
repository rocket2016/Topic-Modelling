from preprocessing.preprocessing_texts import TextPreprocessor
from preprocessing.preprocessing_utils import Preprocessing
from lda.train.lda import Model
from lda.utils import Utils
import matplotlib.pyplot as plt
if __name__ == '__main__':
    preprocessor = TextPreprocessor('ukrainian_texts', 'stop.txt')
    texts_dictionary, names_dictionary= Preprocessing.read_texts_in_dictionaries('ukrainian_texts')
    model = Model(50, alpha=0.8, betta=0.8)
    # model.train(texts_dictionary.values())
    corpus, id2word, text_matrix = model.create_corpus(texts_dictionary.values())
    model_list, coherence_values = Utils.optimaze_topics_amount(id2word, corpus, text_matrix, 10, model.alpha, model.betta, start=2, step=1)

    limit = 10;
    start = 2;
    step = 1;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

