import numpy as np

class TFIDF:
    def __init__(self):
        self.min_df = 30
        self.df = 0
        self.tf = 0
        self.corpus_size = 0
        self.total_number_of_words_in_each_text = {}
        self.tf_scores_of_each_word_in_each_text = {}
        self.document_count_for_each_word= {}
        self.tf_idf_scores_of_each_word_in_each_text = {}
        self.inverse_document_count_for_each_word = {}
        self.tf_idf_values = []
        self.tf_idf_words = []
        self.all_unique_words_in_corpus = set()

    def calculate_values(self,text_array):
        # will calculate tf scores of each word in each text directly
        # will calculate document count for each word indirectly
        for index,text in enumerate(text_array):
            text = text.split()
            self.total_number_of_words_in_each_text[index],words=self.get_total_number_of_terms_for_text(text)
            # get number of times each word appears in single text
            for word in text:
                words[word] += 1
            # normalize with total number of words
            for word in words:
                words[word] = words[word] / self.total_number_of_words_in_each_text[index]
            self.tf_scores_of_each_word_in_each_text[index] = words
        return self.tf_scores_of_each_word_in_each_text
    
    def calculate_idf_for_each_word(self):
        # document count becomes idf
        for key in self.document_count_for_each_word:
            self.inverse_document_count_for_each_word[key] = np.log10(self.corpus_size / self.document_count_for_each_word[key])
        return

    
    def fit(self,text_array):
        # calculate idf and idf * tf
        self.corpus_size = len(text_array)
        print(f"corpus is{self.corpus_size}")
        self.calculate_values(text_array)
        self.calculate_idf_for_each_word()
        for nth_text in self.tf_scores_of_each_word_in_each_text:
            l = []
            temp_dict = {}
            temp = []
            for each_unique_word in self.document_count_for_each_word:
                if self.document_count_for_each_word[each_unique_word] > 20:
                    if each_unique_word in self.tf_scores_of_each_word_in_each_text[nth_text]:
                        temp_dict[each_unique_word] = self.tf_scores_of_each_word_in_each_text[nth_text][each_unique_word] * self.inverse_document_count_for_each_word[each_unique_word]
                    else:
                        temp_dict[each_unique_word] = 0
                    temp.append(temp_dict[each_unique_word])
                    
            self.tf_idf_scores_of_each_word_in_each_text[nth_text] = temp_dict 
            self.tf_idf_values.append(temp)          
        return 

    def test_values(self):
        return np.array(self.tf_idf_values)
    
    def return_tf_idf_values(self):
        return np.array(self.tf_idf_values)
        

    def get_total_number_of_terms_for_text(self,text):
        text_set = set(text)
        self.all_unique_words_in_corpus = self.all_unique_words_in_corpus.union(text_set)
        # update the number of documents the word appears in
        for word in text_set:
            self.document_count_for_each_word[word] = 1 + self.document_count_for_each_word.get(word,0)
        text_set = {word: 0 for word in text_set}
        return len(text_set),text_set

