from preprocessing.preprocessing_utils import Preprocessing

class TextPreprocessor:
    def __init__(self, directory, stop_words_file):
        texts_dictionary, names_dictionary = Preprocessing.read_texts_in_dictionaries(directory)
        self.texts_dictionary = texts_dictionary
        self.names_dictionary = names_dictionary
        self.stop_words_list = Preprocessing.read_stop_words_in_list(stop_words_file)

    def preprocess(self):
        tockens_dictionary = Preprocessing.append_texts_tockens_in_dictionary(self.texts_dictionary)
        stems_dictionary = Preprocessing.stemmatize_texts_tockens(tockens_dictionary)
        stop_stems_list = Preprocessing.stemmatize_stop_words_list(self.stop_words_list)
        preprocessed_texts_list = Preprocessing.stop_words_stems_dictionary_cleaning(stems_dictionary, stop_stems_list)
        return preprocessed_texts_list