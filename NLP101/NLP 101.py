
# coding: utf-8

# # Introduction to ContextEdge (Python package)
# <hr>
# 
# ContextEdge is **Deloitte's integrated toolkit for Natural Language Processing (NLP)** that delivers sophisticated text analytics, advanced context analytics, and leading edge semantic signals analysis solutions for our clients, harnessing potentially valuable insights otherwise undiscovered. The codebase is meant to be used by data scientists as a portable set of accelerators that represents the 70-80% initial solution you need to begin analyzing text for client projects.
# 
# It is designed to help you as a data scientist or developer quickly ingest and analyze a set of documents **_on day one_** so you can spend less time wrangling texxt data and more time making your models more insightful! It uses commons packages such as `sklearn`, `gensim`, and `nltk` that you are familiar with, but packaged into smart workflows as a series of functions that enable you to quickly execute the typical NLP project.
# <hr>
# 
# ##### This notebook will walk you through the basics of the `ContextEdge` codebase, get you comfortable with the architecture, and show you how to customize functions for your needs on client projects!

# <hr> 
# ## Getting Started

# Before beginning this notebook, you should have the ContextEdge package installed. If not, please refer to the **CTxE 001** setup notebook for instructions.

# In[ ]:


# import packages
import os
import pandas as pd
import ContextEdge as ctxe


# In[ ]:


# check our working directory
os.getcwd()


# <hr>
# ## The `Corpus` class
# 
# There are two main classes in `ctxe` codebase -- the `Doc` which is a text collection of a document and the `Corpus`, which is a set of `Doc`s. These classes are located in the central module of `ctxe` -- the `universe` module.
# 
# The `Corpus` class is the main way to analyze a set of `Doc`s. When you instantiate the `Corpus`, you will also pass the set as an argument. ContextEdge handles four types of document sets:
# 
# 1. A **path** (local or url) that points to a folder of individual files that represent the collection of documents;
# 2. A **list** of text strings, which could also be a Series or column in a DataFrame;
# 3. A **column of data** in a csv or xlsx file; and
# 4. An **existing CTxE corpus**.

# In[ ]:


# set of documents as a list
corpora = ['The quick brown fox.','Jumps over the lazy dog!', 
           'My dog is lazy and not very quick. He does like to chase foxes.',
           'On my hikes I enjoy jumping over the creek...']


# In[ ]:


# print the corpora
corpora


# Once we have created a set of documents (corpus) we can proceed with instantiating the _Corpus()_ method to an instance variable _corp_.

# In[ ]:


# establish a Corpus class
# Corpus takes the data object as a required option
corp = ctxe.universe.Corpus(corpora)

# returns the ctxe.universe.Corpus object
corp


# The above will yield a location in our local machine's memory where this object is located. We can look at the location of each document and the number of documents in local memory within the corpus method.

# In[ ]:


# corpus attribute holds the list of the n documents you just ingested
print('Number of docs:', len(corp.corpus))
corp.corpus


# <hr>
# ## Doc Methods
# 
# In the slides that accompany this NLP 101 training, we learned about some important concepts that deal with getting the text data in a format that ContextEdge can use. This is called text pre-processing and is the most important part of the NLP practitioner's job. The Corpus class uses the following architecture to pre-process these documents:
# 
#  - Removing stopwords
#  - Lemmatizing
#  - Lowercasing
#  - Removing punctuation
#  - Removing special characters
#  - Creating word tokens

# `Corpus` class uses list-like indexing to retrieve documents from within the corpus.

# In[ ]:


# look at first document
doc1 = corp.corpus[0]
type(doc1)


# In[ ]:


# what can I do with a document?
dir(doc1)


# The attributes to pay attention to are:
# 
# - **tokens**: Returns the document's tokens and token groups
#     - *words*
#     - *bigrams*
#     - *trigrams*
#     - *sentences*
# - **top_n**: Get the top words in the document, by frequency
# 
# 
# - **text**: Returns the text
# - **clean_text**: Returns the processed text, removing punctuation, lowercasing words, lemmatizing
# 
# 
# - **sentiment_label**: Returns the sentiment label for the document (Positive, negative, neutral)
# - **sentiment_map**: Calculates the sentiment score for the document
# 
# - **filename**: Returns the filename
# - **label**: Returns the text label

# In[ ]:


# look at plain text of document 1
doc1.text


# In[ ]:


# processed text
doc1.clean_text


# In[ ]:


# look at the entire sentence
doc1.sentences


# In[ ]:


# look at the word tokens
doc1.words


# In[ ]:


# look at tokenization in different forms
doc1.tokens


# ## Corpus Methods
# 
# Like `Doc` class, the `Corpus` class also has built-in methods to conduct routine analyses. Later, go back and explore these methods.
# 
# - **clean_corpus**: Remove stopwords from the Corpus
# - **cluster_analysis**: Perform unsupervised clustering analysis
# - **document_similarity**: Use compare documents and find similar ones
# - **frequency_analysis**: Return token frequencies
# - **get_most_similar_docs**: Identify similar documents
# - **matrix2df**: Create a dataframe from word matrix  
# - **plot_cluster_analysis**: Plot the cluster analysis
# - **plot_frequencies**: Plot the frequency analysis
# - **polarity**: Label documents as positive, negative, neutral sentiment
# - **reduce_dim**: Perform dimensionality reduction
# - **sentiment**: Return sentiment analysis for each document
# - **top_n**: Get the top most frequent words

# <hr>
# ### Frequency analysis
# 
# Use `frequency_analysis()` to get counts of words across the `tokens` in the `Corpus`, using `sklearn` to return the `FreqDist()` of the words in the `Corpus`. By default, it calculates monogram, bigrams, and trigrams. `frequency_analysis()` can also be applied to a single `Doc`

# In[ ]:


# get counts of monograms, bigrams, and trigrams across all the docs in the corpus
corp.frequency_analysis()


# Notice that after you run `frequency_analysis()`, a message prints indicating that a new `Corpus` attribute was created. We can now call `freq_distros` and objects within it.

# In[ ]:


# call 'bigrams' from freq_distros
corp.freq_distros['bigrams']


# <hr>
# ### Most frequent words
# 
# Use `top_n()` to get the top `words` in the `Corpus` with the highest frequency. `top_n` can also be applied to a single `Doc`

# In[ ]:


# get the top words in the corpus
# default is to return top 10 monograms, bigrams, and trigrams, and the frequency count of each
top_words = corp.top_n()

# get just to single top words and their counts
top_words['monograms']


# In[ ]:


# get the top 3 words in the last document, and the frequency count of each
doc4 = corp.corpus[-1]
doc4.top_n(3)['monograms']


# <hr>
# ### Plotting word frequencies
# 
# `top_n()` and `frequency_analysis()` produced lists of tokens and their frequencies. That's often not the best way to display that information, especially to a client or when we have a non-trivial number of tokens. 
# 
# The `Corpus` class has a built-in method for plotting these results, `plot_frequencies()`, that will return column charts of the token frequencies.

# In[ ]:


# We can also use the plot_frequencies() method to perform this
corp.plot_frequencies()


# <hr> 
# ## Yelp Reviews

# For the following real-world analysis, we will use what we have learned in the ContextEdge `Corpus()` class to conduct a high-level analysis of Yelp reviews. This data was gathered from a previous kaggle.com competition and it is in a CSV file format. 
# 
# Let's try two different methods of importing the text. 
# 
# First, we will use pandas to extract the "Text" column which contains the reviews that we need to put into a list so we can read it into ContextEdge `Corpus()` class as a list of strings.

# In[ ]:


# create dataframe from Yelp reviews file
df = pd.read_csv("Yelp.csv")


# In[ ]:


# inspect the file
df.head()


# In[ ]:


# select column with the review text from dataframe and turn into a list
corpora = list(df["text"])
corpora[0:5]


# If you recall from the earlier process of instantiating the Corpus class, you will remember that we need to use the initialized variable (what we have called "corp" previously) to call methods and attributes. The code below will initialize the `Corpus` class to a "corp" variable, use the corp variable to ingest the corpora (our Yelp review text) and then return the variable as an object of the `Corpus` class.

# In[ ]:


# establish a Corpus class, specifying the data object
corp = ctxe.universe.Corpus(corpora)

# returns the ctxe.universe.Corpus object
type(corp)


# That was cumbersome! It took 3 steps to read data in: 
# 
# - Creating a `dataframe`
# - Extracting the text as a list
# - Creating the `Corpus` object
# 
# Instead, we can specify the data file and the column which contains our text right when we instantiate the Corpus object.

# In[ ]:


# establish a Corpus class, reading in the Yelp dataset, and specifying text_col = 'text'
# 'text' is the name of the column which contains the reviews
corp = ctxe.universe.Corpus("Yelp.csv",text_col='text')


# What we did in 3 lines, we executed in 1 line!

# In[ ]:


# look at first document
review_1 = corp.corpus[0]


# In[ ]:


type(review_1)


# In[ ]:


review_1.text


# The text above is the first review in the Yelp dataset. We can see that there are areas that need to be pre-processed. The pre-processing has already taken place when we instantiated the `Corpus` class. If we call the attributes of the `Corpus` class, then we can see the processed text.

# In[ ]:


# processed text
review_1.clean_text


# In[ ]:


# look at tokenization in different forms
review_1.tokens


# In[ ]:


# get the top words in the corpus
# default is to return top 10 monograms, bigrams, and trigrams, and the frequency count of each
top_words = review_1.top_n()

# get just to single top words and their counts
top_words['monograms']


# In[ ]:


# get the top bigrams in the corpus
top_words = review_1.top_n(5)

# get just to single top words and their counts
top_words['monograms']


# In[ ]:


pd.DataFrame(top_words['monograms'], columns=['words','occurrences']).set_index('words').plot(kind='bar');


# We have done a good job pre-processing the text and analyzing the distribution of words for the first Yelp review. But what if we want to look at all 500 reviews? We will need to run a loop that can iterate through each document. This way, we will be able to use the attributes on the Corpus class that we learned about in the beginning of the lesson.

# In[ ]:


# Now let's look at all the text in the corpus
[doc.text for doc in corp.corpus]


# In[ ]:


# All reviews pre-processed "clean" text
[doc.text for doc in corp.corpus]


# In[ ]:


# Tokens for all reviews
[doc.tokens for doc in corp.corpus]


# In[ ]:


top_words = corp.top_n(n=25)
top_words['monograms']


# In[ ]:


corp.plot_frequencies()


# <hr>
# ## Sentiment analysis
# 
# The `sentiment` module currently performs analysis to classify the sentiment of a sentence in a document based on it's polarity score. This is leveraging the `nltk` implementation, using the trained sentiments from the `opinion_lexicon`.
# 
# Sentiment for an entire `Doc` is calculated using the `sentiment_label()` function, which averages the sentiments from the sentences to determine an overall sentiment at the document level.
# 
# Let's build up to this.

# In[ ]:


# retrieve the first document from the Yelp corpus
doc1 = corp.corpus[0]


# In[ ]:


# analyze the raw sentiment of the each sentence in the first document
doc1.sentiment_map()


# Notice the structure of the output above. The result returns a tuple of:
# 
# - the index of the sentence (0);
# - the text of the sentence ('My wife took me here on my birthday for breakfast and it was excellent.'); and 
# - the sentiment of sentence (1).
# 
# To get the document sentiment, we pass the sentence sentiment map into `sentiment_label()`.

# In[ ]:


# get the overall sentiment 
doc1.sentiment_label(doc1.sentiment_map())


# In[ ]:


# get sentiment labels for the first 10 documents
sentiments = [doc.sentiment_label(doc.sentiment_map()) for doc in corp.corpus[0:10]]
sentiments


# <hr>
# # Looking Ahead
# 
# We talked a lot this session about how to analyze text at a basic level: describing the words that comprise the document; the pairs or tuples of words; and whether document is positive or negative based on what words appear. This is only scraping the tip of the iceberg of both ContextEdge and Natural Language Processing.
# 
# In ContextEdge 201, we move from describing the words to analyzing them using more powerful tools, like clustering and topic modeling. 
# 
# For example, instead of describing the most frequent words in a document or in a corpus, we examined which words appear in which documents, which words are unique to certain topics, and which words are more likely to appear together. Let's take a quick look.

# In[ ]:


options = dict(lowercase=False,use_idf=True)
tf_idf = corp.vectorize(vectorizer='tfidf',vec_options=options)


# In[ ]:


df = corp.vec_matrix
words=corp.words
pandas_df = corp.matrix2df(df,words)
pandas_df.head()


# In[ ]:


print('text: ',corp.corpus[9].text,'\n\n clean text: ',corp.corpus[9].clean_text)


# In[ ]:


pandas_df[9:10].transpose().sort_values('Doc9',ascending=False)[:10]


# In[ ]:


corp.cluster_analysis()


# In[ ]:


corp.cluster_label_desc


# In[ ]:


corp.cluster_labels[9]


# In[ ]:


corp.plot_cluster_analysis()


# # Exercises
# 
# Using the provided _Literature_ dataset, complete the below exercises to conduct an exploratory analysis.

# ## Exercise 1

# In[ ]:


# Create a corpora and instantiate a Corpus class
books = ctxe.universe.Corpus("C:/Users/mschillawski/Documents/CTxE/ctxe-py-training/ContextEdge 101/literature.csv",
                             text_col='text')
# Print the first document of the Corpus
books.corpus[0].text


# In[ ]:


# how many words are in your corpus? (hint: print the words)
len(books.frequency_analysis()['monograms'])


# In[ ]:


# print the clean text 
[book.clean_text for book in books.corpus]


# In[ ]:


# print the bigrams of your corpus
books.freq_distros['bigrams']


# ## Exercise 2
# 
# Find the 10 top occuring words in your corpus and plot them

# In[ ]:


# get the top words in the corpus
books.top_n()

# get just to single top words and their counts
top_words = books.top_n(n=25)

# create plot of these words and their counts
pd.DataFrame(top_words['monograms'], columns=['words','occurances']).set_index('words').plot(kind='bar');


# ## Exercise 3
# 
# Return a sentiment score from each document

# In[ ]:


# get sentiment labels for each document
sentiments = [book.sentiment_label(book.sentiment_map()) for book in books.corpus]
sentiments


# In[ ]:


# plot the sentiment distribution of the documents in the corpus

sentiment = {feeling:[sentiments.count(feeling)] for feeling in list(set(sentiments))}
sentiment = pd.DataFrame.from_dict(sentiment,orient='index',columns=['count'])
sentiment.plot(kind='bar')
