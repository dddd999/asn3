from nltk.corpus import stopwords, brown
from gensim import corpora,models, similarities
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.test.utils import common_texts
from gensim.utils import simple_preprocess
from gensim.matutils import softcossim

def W2VH():
   docbrown = ""
   for w in  brown.words(categories='mystery'):
      docbrown += str(w.lower().split())

   docbrown1,docbrown2 = docbrown[:int(len(docbrown)/2)], docbrown[int(len(docbrown)/2):]

   stop_words = stopwords.words('english')
   docbrown1 = [w for w in docbrown1 if w not in stop_words]
   docbrown2 = [w for w in docbrown2 if w not in stop_words]

   documents = [docbrown1, docbrown2]
   dictionary = corpora.Dictionary(documents)

   docbrown1 = dictionary.doc2bow(docbrown1)
   docbrown2 = dictionary.doc2bow(docbrown2)
   
   model = Word2Vec(common_texts, size=20, min_count=1)
   termsim_index = WordEmbeddingSimilarityIndex(model.wv)
   similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)

   similarity = similarity_matrix.inner_product(docbrown1, docbrown2, normalized=True)
   print('= %.4f' % similarity)
    
def W2V():
   docbrown1 = ""
   for w in  brown.words(categories='mystery'):
      docbrown1 += str(w.lower().split())

   docbrown2 = ""
   for w in  brown.words(categories='science_fiction'):
      docbrown2 += str(w.lower().split())

   stop_words = stopwords.words('english')
   docbrown1 = [w for w in docbrown1 if w not in stop_words]
   docbrown2 = [w for w in docbrown2 if w not in stop_words]

   documents = [docbrown1, docbrown2]
   dictionary = corpora.Dictionary(documents)

   docbrown1 = dictionary.doc2bow(docbrown1)
   docbrown2 = dictionary.doc2bow(docbrown2)

   model = Word2Vec(common_texts, size=20, min_count=1)
   termsim_index = WordEmbeddingSimilarityIndex(model.wv)
   similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)

   similarity = similarity_matrix.inner_product(docbrown1, docbrown2, normalized=True)
   print('= %.4f' % similarity)

def TFIDFH():
    document1 = []
    for w in  brown.words(categories='mystery'):
        document1.append(w.lower())

    B = document1[:len(document1)//2]
          
    doc = ""
    for w in  brown.words(categories='mystery'):
        doc += str(w.lower())

    C,D = doc[:int(len(doc)/2)], doc[int(len(doc)/2):]

    stoplist = set('for a of the and to in - , is'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in B]
    
    texts[0] = [text.replace(',','') for text in texts[0]]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

    vec_bow = dictionary.doc2bow(D.lower().split())

    vec_lsi = lsi[vec_bow] 
    index = similarities.MatrixSimilarity(lsi[corpus]) 

    sims = index[vec_lsi] # perform a similarity query against the corpus
    print("results: ",sims)

    bow1 = document1
    bow2 = doc

    wordSet = set(bow1).union(set(bow2))

    wordDict1 = dict.fromkeys(wordSet, 0)
    wordDict2 = dict.fromkeys(wordSet, 0)

    for word in bow1:
        wordDict1[word]+=1
        
    for word in bow2:
        wordDict2[word]+=1


    def computeTF(wordDict, bow):
        tfDict = {}
        bowCount = len(bow)
        for word, count in wordDict.items():
            tfDict[word] = count / float(bowCount)
        return tfDict

    tfBow1 = computeTF(wordDict1, bow1)
    tfBow2 = computeTF(wordDict2, bow2)

    def computeIDF(docList):
        import math
        idfDict = {}
        N = len(docList)
    
        idfDict = dict.fromkeys(docList[0].keys(), 0)
        for doc in docList:
            for word, val in doc.items():
                if val > 0:
                    idfDict[word] += 1
    
        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / float(val))
        
        return idfDict

    idfs = computeIDF([wordDict1,wordDict2])
    print("IDF")
    print(idfs)
    def computeTFIDF(tfBow, idfs):
        tfidf = {}
        for word, val in tfBow.items():
            tfidf[word] = val*idfs[word]
        return tfidf

    tfidfBow1 = computeTFIDF(tfBow1, idfs)
    tfidfBow2 = computeTFIDF(tfBow2, idfs)

    print("TF-IDF Document1: ")
    print(tfidfBow1)
    print("TF-IDF Document2: ")
    print(tfidfBow2)
    
def TFIDFH1():   
   docbrown = ""
   for w in  brown.words(categories='mystery'):
      docbrown += str(w.lower().split())

   docbrown1,docbrown2 = docbrown[:int(len(docbrown)/2)], docbrown[int(len(docbrown)/2):]

   documents = [docbrown1,docbrown2]

   mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
   corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]

   tfidf = models.TfidfModel(corpus, smartirs='lfn')

   tfidf = models.TfidfModel(corpus)
   corpus_tfidf = tfidf[corpus]
   index = similarities.MatrixSimilarity(tfidf[corpus])
   cossim = index[corpus_tfidf]

   print ("= COS TF-IDF: ",cossim)
  
def TFIDF1():
   docbrown1 = ""
   for w in brown.words(categories='mystery'):
      docbrown1 += str(w.lower().split())

   docbrown2 = ""
   for w in brown.words(categories='science_fiction'):
      docbrown2 += str(w.lower().split())

   documents = [docbrown1,docbrown2]

   mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
   corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]

   tfidf = models.TfidfModel(corpus, smartirs='lfn')

   tfidf = models.TfidfModel(corpus)
   
   corpus_tfidf = tfidf[corpus]
   index = similarities.MatrixSimilarity(tfidf[corpus])
   cossim = index[corpus_tfidf]

   print ("= COS TF-IDF: ", cossim)
    
print("TF-IDF first-half Mystery  vs. second-half Mystery:")
TFIDFH1()
print("TF-IDF Myster vs Science-Fiction:")
TFIDF1()
print("W2V Cosine Mystery vs. Science-Fiction:")
W2V()
print("W2V Cosine first-half Mystery vs. second-half Mystery:")
W2VH()
