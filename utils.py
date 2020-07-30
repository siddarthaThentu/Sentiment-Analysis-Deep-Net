import string
import re
import os
import nltk

nltk.download("twitter_samples")
nltk.download("stopwords")

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords,twitter_samples

tweet_tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)

stopwords_english = stopwords.words("english")

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def process_tweet(tweet):
    
    tweet = re.sub("\$\w*",'',tweet)
    tweet = re.sub("^RT[\s]+",'',tweet)
    tweet = re.sub("https?:\/\/.*[\r\n]*",'',tweet)
    tweet = re.sub("#",'',tweet)
    
    tweet_tokens = tweet_tokenizer.tokenize(tweet)
 
    tweets_clean=[]
    
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
     
    return tweets_clean

def load_tweets():
    all_positive_tweets = twitter_samples.strings("positive_tweets.json")
    all_negative_tweets = twitter_samples.strings("negative_tweets.json")
    return all_positive_tweets,all_negative_tweets

class Layer(object):
    
    def __init__(self):
        self.weights = None
        
    def forward(self,x):
        raise NotImplementedError
    
    def init_weigths_and_state(self,input_signature,random_key):
        pass
    
    def init(self,input_signature,random_key):
        self.init_weights_and_state(input_signature,random_key)
        return self.weights
    
    def __call__(self,x):
        return self.forward(x)