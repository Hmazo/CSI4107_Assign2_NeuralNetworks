# import requests
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import math
import re
from autocorrect import Speller
#import pytrec_eval

#assignment 2 method 2
import warnings
warnings.filterwarnings('ignore')
#for our word embeddings
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
model_glove_twitter = api.load("glove-twitter-25")


# create a vocab dict
# (word) : (df, list of dictionaries[documentid:(string of text, tf),...])
# Grab line
# Lower case it (keep original though!)
# Take away "'" because of can't , aint, isn't I'm
# take away punctuation
# remove stop words, then tokenize it, then go through and add to the dictionary the words and strings
# if exists the word added, then increase by 1 (if its only a new document) and the tf should be updated as well
def preProssess(filename):

    # Local get for text file
    file = open(filename, "r", encoding='utf-8')
    # Use this to read file content as a stream:
    fullText = file.read()
    sentences = fullText.split('\n')
    spell = Speller(fast=True)

    stop_words = set(stopwords.words('english'))
    Documents = {}
    vocab = {}
    for i in range(len(sentences)):
        # split by tab
        currSentenceTuple = sentences[i].split('\t')
        # [docid, setence]
        # the key is the tweetid and the value is the tweet text
        # (fill sentence, dictionary of words and their weights, length)
        Documents[currSentenceTuple[0]] = (currSentenceTuple[1], {}, 0)
        # start the preprocessing here
        currSentenceValue = currSentenceTuple[1]
        # lower case
        currSentenceValue = currSentenceValue.lower()
        # remove URLS
        currSentenceValue = re.sub(r'http\S+', '', currSentenceValue)
        # create our tokenizer that will also remove punctuation
        tokenizer = RegexpTokenizer(r'\w+')
        # removing the I'm , can't to Im and cant
        currSentenceValue = currSentenceValue.replace("'", "")

        #autocorrect spelling mistakes
        currSentenceValue = spell(currSentenceValue)

        # tokenize here
        currSentenceValue = tokenizer.tokenize(currSentenceValue)
        # remove stop words
        porterStemmer = PorterStemmer()
        currSentenceValue = [porterStemmer.stem(w) for w in currSentenceValue if not w in stop_words]

        # finished preprossessing tweet
        # send the preprocessed tweet to be indexed
        (vocab, Documents) = indexing(currSentenceValue, currSentenceTuple, Documents, vocab)

    tf_max = 0
    for word in vocab:
        if vocab[word][0] > tf_max:
            tf_max = vocab[word][0]

    numOfDocs = len(Documents)
    for docid in Documents:
        length = 0
        for wordsInDoc in Documents[docid][1]:
            df_i = vocab[wordsInDoc][0]
            idf = math.log((numOfDocs / df_i), 2)
            tf_ij = Documents[docid][1][wordsInDoc] / len(Documents[docid][1])
            w_ij = tf_ij * idf
            Documents[docid][1][wordsInDoc] = w_ij
            length += w_ij ** 2
        (doc, sentence, l) = Documents[docid]
        Documents[docid] = (doc, sentence, math.sqrt(length))
    return (vocab, Documents)


def indexing(currSentenceValue, currSentenceTuple, Documents, vocab):
    # begining indexing
    # performing stemming in this assignment (apparently more efficient)
    # porterStemmer = PorterStemmer()

    # here  we are grabing our dictionary of
    # (currSentenceTuple[1],{},0)     docid = currSentenceTuple[0]
    (fullSentence, dictOfWords, length) = Documents[currSentenceTuple[0]]
    for stemword in currSentenceValue:
        # stemword = porterStemmer.stem(word)
        # check if its a number we dont need this
        if stemword not in dictOfWords:
            # tf
            dictOfWords[stemword] = 1
            Documents[currSentenceTuple[0]] = (fullSentence, dictOfWords, length)
        else:
            dictOfWords[stemword] = dictOfWords[stemword] + 1
            Documents[currSentenceTuple[0]] = (fullSentence, dictOfWords, length)
        if stemword not in vocab:  # if it's not in our vocab then we set (df=1, new dict with tweetid as key and tf=1)
            vocab[stemword] = (1, {currSentenceTuple[0]: 1})
        else:  # word is already in our vocab
            (df, docs) = vocab[stemword]
            if currSentenceTuple[0] in docs:
                docs[currSentenceTuple[0]] = docs[currSentenceTuple[0]] + 1
                vocab[stemword] = (len(docs), docs)
            else:
                docs[currSentenceTuple[0]] = 1
                vocab[stemword] = (len(docs), docs)
    return (vocab, Documents)
       
def queryResults(queryString, vocabDict, documents, numberOfRowsForResults):
    stop_words = set(stopwords.words('english'))
    scores = {}
    N = len(documents)
    queryString = queryString.lower()
    #queryStringExpansion = queryExpansionMethod(model_glove_twitter,queryString)
    queryStringExpansion = queryString
    # create our tokenizer that will also remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    # removing the I'm , can't to Im and cant
    queryString = queryString.replace("'", "")
    # tokenize here
    queryString = tokenizer.tokenize(queryString)
    # remove stop words
    porterStemmer = PorterStemmer()
    queryString = [porterStemmer.stem(w) for w in queryString if not w in stop_words]

    # we are collecting the weights for the query string and it's length
    weightsForQuery = {}
    lengthOfQuery = 0
    for stemword in queryString:
        if stemword.isnumeric():
            continue
        #adding check here so see if the stem word is actually in our vocab. If it's not then we can simply skip it
        if stemword not in vocabDict:
            continue
        # docsFoundForStemWord = vocabDict[stemword]
        # calculate weight for query word i
        df_i = vocabDict[stemword][0]
        tf_iq = queryString.count(stemword) / len(queryString)
        idf = math.log((N / df_i), 2)
        w_iq = (0.5 + 0.5 * tf_iq) * idf
        if stemword not in weightsForQuery:
            weightsForQuery[stemword] = w_iq
            lengthOfQuery += w_iq ** 2

    # we now have the length of the query vector and a dict of weights w_iq
    lengthOfQuery = math.sqrt(lengthOfQuery)

    # print(weightsForQuery)

    for word in weightsForQuery:
        docsFoundForStemWord = vocabDict[word][1]
        for doc in docsFoundForStemWord:
            scores[doc] = cosineCalculator(doc, documents, lengthOfQuery, weightsForQuery)


    arrayOfSortedScoresTuples = sorted(scores.items(), key=lambda x: x[1], reverse=True )
    #here we add a dictionary that will store the documents and their new scores on the query expansion
    arrayOfSortedScoresTuplesExpanded = {}
    for i in range(len(arrayOfSortedScoresTuples)):
        docId = arrayOfSortedScoresTuples[i][0]
        originalScore = arrayOfSortedScoresTuples[i][1]
        docSentence = documents[docId][0] #get sentence
        #get the tokens in our twitter embedding model 
        tokens_1=[t for t in docSentence.split() if t in model_glove_twitter]
        tokens_2=[t for t in queryStringExpansion.split() if t in model_glove_twitter]
        cosine=0
        if (len(tokens_1) > 0 and len(tokens_2)>0):
            cosine=model_glove_twitter.n_similarity(tokens_1,tokens_2)
            #take the average of both scores!
            newScoreAvg = (originalScore+cosine)/2
            #store the score with the document
            arrayOfSortedScoresTuplesExpanded[docId]=newScoreAvg
    #sort by highest value!
    arrayOfSortedScoresTuplesExpanded= sorted(arrayOfSortedScoresTuplesExpanded.items(), key=lambda x: x[1], reverse=True)  
    return arrayOfSortedScoresTuplesExpanded[:numberOfRowsForResults]


def cosineCalculator(documentid, documents, lengthOfQuery, weightsForQuery):
    (sentence, weights, lengthOfDoc) = documents[documentid]
    numerator = 0
    for queryWord in weightsForQuery:
        if queryWord in documents[documentid][1]:
            numerator += documents[documentid][1][queryWord] * weightsForQuery[queryWord]
    return (numerator) / (lengthOfDoc * lengthOfQuery)


# Get request! If needed to access non local txt file
# file1 = requests.get("http://www.site.uottawa.ca/~diana/csi4107/A1_2021/Trec_microblog11.txt")
# line = file1.text
(vocab, Documents) = preProssess("Trec_microblog11.txt")
# w_iq = (0.5 + 0.5 tf_iq)∙idf_i      this is gonna be used for our queries!
# print(Documents['32529490546532352'])
# print results -> [(docid, rank score), ...]

'''
results = queryResults("Afghanistan bombs terrorists", vocab, Documents, 100)
print(len(results))

# Documents['docid'] ->(sentence, dicitionaryOfWeights, lengthOfVector)

print(Documents[results[0][0]][0])  # actual tweet
for i in range(len(results)):
    print(results[i])  # print results -> [(docid, rank score), ...]
'''

print("What is the run id: ")
run_id = input()  # id chosen for specific run

# have to output doc format topic_id Q0 docno rank score tag
# queries from test file
queriesDoc = open("test_queries_49.txt", "r")
queriesLst = []
qcnt = 0  # query iterat counter
f = open("results_file_method2.txt", "w")

query = ["topic_id", "query title"]
for line in queriesDoc:
    if "<num>" in line:  # looking for <num> identifier for topic_id
        query[0] = line.replace("<num> Number: ", "").replace(" </num>", "").replace("MB00","")

    if "<title>" in line:  # actual query
        query[1] = line.replace("<title> ", "").replace(" </title>", "")
        queriesLst.append(query)
        query = ["", ""]

# queriesLst[][] =[][topic_id, doc_no, querty title]
for x in range(49):  # iterate 49 queries
    results = queryResults(queriesLst[x][1], vocab, Documents, 1000)
    for i in range(min(len(results), 1000)):  # top 1000 results for each query if that many
        topic_id = queriesLst[x][0].rstrip('\n')
        Q0 = "Q0"
        doc_no = results[i][0]  # results -> [(docid, rank score), ...]
        # tweet id
        rank = i  # 1 is highest rank\
        score = results[i][1]  # computed degree of match between the segment and the topic

        f.write(str(x+1) + " " + Q0 + " " + doc_no + " " + str(rank) + " " + str(score) + " " + run_id + "\n")

f.close

#Line to compare with TREC results
#pytrec_eval._RelevanceEvaluator.evaluate() 


