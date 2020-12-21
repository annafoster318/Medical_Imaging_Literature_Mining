from Bio import Entrez
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re
import numpy as np


articles = []  # where all pubmed articles will be stored
start = 0
interval = 10  # how many articles we will get in one loop
max = interval
while max <= 50:  # loop so can get all results in timely manner (retmax in esearch was 100,000)
    def search(query):
        Entrez.email = 'anfost17@stlawu.edu'
        handle = Entrez.esearch(db='pubmed',
                                sort='relevance',
                                usehistory= 'y',
                                retstart= start,
                                retmax= interval,
                                retmode='xml',
                                datetype='pdat',
                                mindate='2017',
                                term=query)
        results = Entrez.read(handle)
        return results

    print("Accessing articles:", start, "to", max)  # this is here as a check

    def fetch_details(id_list):
        ids = ','.join(id_list)
        Entrez.email = 'anfost17@stlawu.edu'
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)
        return results

    # search the pubmed database
    if __name__ == '__main__':
        results = search('diagnostic imaging')  # put search term here
        id_list = results['IdList']
        papers = fetch_details(id_list)
        index = 1
        for i, paper in enumerate(papers['PubmedArticle']):
            # print("%d) %s" % (start+index, paper['MedlineCitation']['Article']['ArticleTitle'])) #gives article titles
            # add each pubmed article abstract to the total articles list
            articles.append(paper['MedlineCitation']['Article']['Abstract']['AbstractText'])
            index += 1
        # Pretty print the first paper in full to observe its structure
        # import json
        # print(json.dumps(papers[0], indent=2, separators=(',', ':')))
        # return articles
    # a lot of above code is from: https://marcobonzanini.com/2015/01/12/searching-pubmed-with-python/

    # move to the next 10 in the list
    start += interval
    max += interval


# and remove stop words
stopW = open('stopwords.txt', 'r')
# convert stop words text to list of stop words
stopWords = stopW.read().replace("\n", " ")
stopWords_list = stopWords.split()
stopW.close()


# the articles are currently in a list of lists of strings
# modify to just list of strings
newArticlesList = []
# print("The total article list:")
for i, article in enumerate(articles):
    # print("%d) %s" % (i + 1, str(article)))
    newArticlesList.append(' '.join(article))
# print(newArticlesList)


# create bi-grams (if two words appear together >100 times, join them with a _)
# this will be done through turning the articles into a list of words and creating a dictionary
countDict = {}
count = 1
articleString = ""
for i, article in enumerate(newArticlesList):
    articleString = articleString + " " + article.lower()
# print(articleString)
wordsList = articleString.split()  # this is a list of all the words in the abstracts
# we want to remove stop words from this list now
for word in stopWords_list:
    for articleWord in wordsList:
        if articleWord == word:
            wordsList.remove(articleWord)
# print(wordsList)
punctuation = string.punctuation.replace('-', '')  # we'll remove punctuation too, but keep hyphenated words
table = str.maketrans('', '', punctuation)
strippedWords = [w.translate(table) for w in wordsList]
# print(strippedWords)
# print(wordsList)
for i, word in enumerate(strippedWords):
    if (i+1) < len(strippedWords):
        word2 = strippedWords[i+1]
        # check if this key already exists
        if (word + " " + word2) in countDict:
            count += 1
            countDict[word + " " + word2] = count
        else:
            countDict[word + " " + word2] = count
    else:
        break  # means the end of the list of words has been reached
# a dictionary has been made where the key is the two words,
# and the value is the number of times those words appear together
# print(countDict)

# now for the words with values over 100, join them with a _ in the abstracts
bigramsList = []
for words, occurence in countDict.items():
    if occurence > 100:
        bigramsList.append(words)
print(bigramsList)

articlesEditList = []
finalArticles = []
for i, article in enumerate(newArticlesList):
    articleLower = article.lower()
    articlesEditList.append(articleLower)

for i, article in enumerate(articlesEditList):
    articleWords = article.split()
    for x, word in enumerate(articleWords):
        if (x+1) < len(articleWords):
            word2 = articleWords[x+1]
            if (word + " " + word2) in bigramsList:
                articleWords[x] = (word + "_" + word2)
                del articleWords[x+1]
        else:
            newArticle = (' '.join(articleWords))
            finalArticles.append(newArticle)
            break  # we're at the end of the article
# print(finalArticles)

# --------------------------------------------------------

# start topic modeling with scikit learn
# using both NMF and LDA
# following code written using https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
# and https://towardsdatascience.com/improving-the-interpretation-of-topic-models-87fd2ee3847d

no_topics = 10
no_top_words = 20
no_top_docs = 5

def display_topics(H, W, vectorizer, documents, no_top_words, no_top_docs):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_docs]
        for doc_index in top_doc_indices:
            print(documents[doc_index])

# create NMF method
n_vectorizer = TfidfVectorizer(min_df=5, max_df=.9, stop_words='english',
                             lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
n_data_vectorized = n_vectorizer.fit_transform(newArticlesList)

# run NMF method
nmf_model = NMF(n_components=no_topics)
nmf_W = nmf_model.fit_transform(n_data_vectorized)
nmf_H = nmf_model.components_

# create the LDA method
l_vectorizer = CountVectorizer(min_df=5, max_df=.9, stop_words='english',
                             lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
l_data_vectorized = l_vectorizer.fit_transform(newArticlesList)

# run LDA method
lda_model = LatentDirichletAllocation()
lda_W = lda_model.fit_transform(l_data_vectorized)
lda_H = lda_model.components_

print("=" * 20)
print("The NMF Method:")
display_topics(nmf_H, nmf_W, n_vectorizer, finalArticles, no_top_words, no_top_docs)

print("=" * 20)
print("The LDA Method:")
display_topics(lda_H, lda_W, l_vectorizer, finalArticles, no_top_words, no_top_docs)

