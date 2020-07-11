import os
import re
import fnmatch
import numpy as np
class Preprocessing:
    @staticmethod
    def read_texts_in_dictionaries(directory):
        i = 0
        d = {}
        d_name = {}
        for file_name in os.listdir(directory + '/'):
            if fnmatch.fnmatch(file_name, '*.txt'):
                with open(directory + '/' + file_name, 'r', encoding='utf-8', errors='ignore') as file:
                    data = file.read().replace('\n', '')
                    d["text{0}".format(i)] = re.sub(r'\{[^)]*\}|\,|\.|\»|\«|\—|\(|\)|\s(а|та|в|i|:|й|той|ті|y|\+|\-)\s',
                                                    '', data)
                    d_name["text{0}".format(i)] = file_name
                    i = i + 1
        return d, d_name

    @staticmethod
    def read_stop_words_in_list(file):
        stop_words_list = []
        with open(file, 'r', encoding='utf-8', errors="ignore") as file:
            stop_words_list.append(file.read().split('\n\n'))
        arr = np.array(stop_words_list)
        stop_words_list = arr.flatten().tolist()

        return stop_words_list

    @staticmethod
    def append_texts_tockens_in_dictionary(texts_dictionary):
        import nltk
        d_tockens = {}
        # nltk.download('punkt')
        for key, value in texts_dictionary.items():
            d_tockens.setdefault(key, nltk.word_tokenize(value))
        return d_tockens

    @staticmethod
    def stemmatize_texts_tockens(tockens_dictionary):
        d_stems = {}
        for key, value in tockens_dictionary.items():
            stems = []
            for tocken in value:
                stems.append(UkrainianStemmer.stem_word(tocken))
            d_stems.setdefault(key, stems)
        return d_stems

    @staticmethod
    def stemmatize_stop_words_list(stop_words_list):
        stop_stems_list = []
        for word in stop_words_list:
            stop_stems_list.append(UkrainianStemmer.stem_word(word))
        return stop_stems_list

    @staticmethod
    def stop_words_stems_dictionary_cleaning(stems_dictionary, stop_stems_list):
        d_clean_stems = {}
        for key, value in stems_dictionary.items():
            for stem in stop_stems_list:
                if stem in value:
                    value.remove(stem)
            d_clean_stems.setdefault(key, value)
        texts = list(d_clean_stems.values())

        return texts

    @staticmethod
    def sampling(texts, sample_size):
        samples = []
        for text in texts:
            sample = np.random.choice(text, sample_size)
            samples.append(sample)
        return samples




class UkrainianStemmer:
    __vowel = r'аеиоуюяіїє'  # http://uk.wikipedia.org/wiki/Голосний_звук
    __perfectiveground = r'(ив|ивши|ившись|ыв|ывши|ывшись((?<=[ая])(в|вши|вшись)))$'
    # http://uk.wikipedia.org/wiki/Рефлексивне_дієслово
    __reflexive = r'(с[яьи])$'
    # http://uk.wikipedia.org/wiki/Прикметник + http://wapedia.mobi/uk/Прикметник
    __adjective = r'(ими|ій|ий|а|е|ова|ове|ів|є|їй|єє|еє|я|ім|ем|им|ім|их|іх|ою|йми|іми|у|ю|ого|ому|ої)$'
    # http://uk.wikipedia.org/wiki/Дієприкметник
    __participle = r'(ий|ого|ому|им|ім|а|ій|у|ою|ій|і|их|йми|их)$'
    # http://uk.wikipedia.org/wiki/Дієслово
    __verb = r'(сь|ся|ив|ать|ять|у|ю|ав|али|учи|ячи|вши|ши|е|ме|ати|яти|є)$'
    # http://uk.wikipedia.org/wiki/Іменник
    __noun = r'(а|ев|ов|е|ями|ами|еи|и|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я|і|ові|ї|ею|єю|ою|є|еві|ем|єм|ів|їв|ю)$'
    __rvre = r'[аеиоуюяіїє]'
    __derivational = r'[^аеиоуюяіїє][аеиоуюяіїє]+[^аеиоуюяіїє]+[аеиоуюяіїє].*(?<=о)сть?$'
    __RV = ''

    @staticmethod
    def ukstemmer_search_preprocess(word):
        word = word.lower()
        word = word.replace("'", "")
        word = word.replace("ё", "е")
        word = word.replace("ъ", "ї")
        return word

    @staticmethod
    def s(st, reg, to):
        orig = st
        UkrainianStemmer.__RV = re.sub(reg, to, st)
        return (orig != UkrainianStemmer.__RV)

    @staticmethod
    def stem_word(word):
        word = UkrainianStemmer.ukstemmer_search_preprocess(word)
        if not re.search('[аеиоуюяіїє]', word):
            stem = word
        else:
            p = re.search(UkrainianStemmer.__rvre, word)
            start = word[0:p.span()[1]]
            UkrainianStemmer.__RV = word[p.span()[1]:]

            # Step 1
            if not UkrainianStemmer.s(UkrainianStemmer.__RV, UkrainianStemmer.__perfectiveground, ''):

                UkrainianStemmer.s(UkrainianStemmer.__RV, UkrainianStemmer.__reflexive, '')
                if UkrainianStemmer.s(UkrainianStemmer.__RV, UkrainianStemmer.__adjective, ''):
                    UkrainianStemmer.s(UkrainianStemmer.__RV, UkrainianStemmer.__participle, '')
                else:
                    if not UkrainianStemmer.s(UkrainianStemmer.__RV, UkrainianStemmer.__verb, ''):
                        UkrainianStemmer.s(UkrainianStemmer.__RV, UkrainianStemmer.__noun, '')
            # Step 2
            UkrainianStemmer.s(UkrainianStemmer.__RV, 'и$', '')

            # Step 3
            if re.search(UkrainianStemmer.__derivational, UkrainianStemmer.__RV):
                UkrainianStemmer.s(UkrainianStemmer.__RV, 'ость$', '')

            # Step 4
            if UkrainianStemmer.s(UkrainianStemmer.__RV, 'ь$', ''):
                UkrainianStemmer.s(UkrainianStemmer.__RV, 'ейше?$', '')
                UkrainianStemmer.s(UkrainianStemmer.__RV, 'нн$', u'н')

            stem = start + UkrainianStemmer.__RV
        return stem
