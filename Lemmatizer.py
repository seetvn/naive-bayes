
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import opinion_lexicon
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('words')
# nltk.download('opinion_lexicon')
lemmatizer = WordNetLemmatizer()

def concat_feature_list_into_list(my_list):
  ans = []
  for item in my_list:
    ans.append(''.join(item))
  return ans

def get_ngrams(lemmatized_words, range_):
  total_list = []
  ans = []
  for x in range(1,range_ + 1):
    total_list += list(ngrams(lemmatized_words, x))
  return total_list

def lemmatize_text(text):
  stop_words = set(stopwords.words('english'))
  t = text.split('.')
  text = []
  lemmatized_words = []

  for t_ in t:
     text.append(''.join([char for char in t_ if char not in string.punctuation]))
  # print(text)
  # text = ''.join([char for char in text if char not in string.punctuation])
  for sentence in text:
    words = nltk.word_tokenize(sentence)
    pos_tags = pos_tag(words)
    ind = 0
    for word, pos in pos_tags:
      if word.lower() not in stop_words and (word == word.lower() or (word !=word.lower() and ind == 0)):
          if pos.startswith('V'):  # Check if the word is a verb
              lemma = lemmatizer.lemmatize(word, pos='v')  # Lemmatize verb
          elif pos.startswith('N'):  # Check if the word is a noun
              lemma = lemmatizer.lemmatize(word, pos='n')  # Lemmatize noun
          elif pos.startswith('J'):  # Check if the word is an adjective
              lemma = lemmatizer.lemmatize(word, pos='a')  # Lemmatize adjective
          elif pos.startswith('R'):  # Check if the word is an adverb
              lemma = lemmatizer.lemmatize(word, pos='r')  # Lemmatize adverb
          elif pos.startswith('U'):
            lemma = lemmatizer.lemmatize(word, pos='r')  # Lemmatize interjection
          else:
            lemma = lemmatizer.lemmatize(word)  # Default to lemmatizing as a noun
          if lemma not in lemmatized_words:
              lemmatized_words.append(lemma)
      ind = 1

  return lemmatized_words