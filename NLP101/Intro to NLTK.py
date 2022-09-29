"""°°°
## Intro to NLTK
°°°"""
# |%%--%%| <lS7NksGe21|F6qXgfxzIs>

import nltk 
#Project Gutenburg - open source database for books and text
nltk.download('gutenberg') #download gutenburg

# |%%--%%| <F6qXgfxzIs|JfS4fpZJiD>

nltk.corpus.gutenberg.fileids()

# |%%--%%| <JfS4fpZJiD|W8lOXj6DXr>

#Load the book moby dick
md = nltk.corpus.gutenberg.words('melville-moby_dick.txt')

# |%%--%%| <W8lOXj6DXr|nx8SIfWWyA>

md

# |%%--%%| <nx8SIfWWyA|mWIBd1x5x9>

md_set =set(md)

# |%%--%%| <mWIBd1x5x9|6op4Y3gTIR>

len(md_set)

# |%%--%%| <6op4Y3gTIR|qyIbt7kvuB>

len(md) / len(md_set)

# |%%--%%| <qyIbt7kvuB|hYmIFieYPH>

md_sents = nltk.corpus.gutenberg.sents('melville-moby_dick.txt')

# |%%--%%| <hYmIFieYPH|yF5dSATkst>

len(md) / len(md_sents)

# |%%--%%| <yF5dSATkst|NI8U7PNpTP>
"""°°°
## Example-Words Per Sentence Trends
°°°"""
# |%%--%%| <NI8U7PNpTP|zDXFQQGfmi>

from nltk.corpus import inaugural
nltk.download('inaugural')

# |%%--%%| <zDXFQQGfmi|jwS8juoLuB>

inaugural.fileids()

# |%%--%%| <jwS8juoLuB|SGR3TBLwk0>

speeches = inaugural.fileids()
for speech in speeches:
    speech_sents = inaugural.sents(speech)
    speech_words = inaugural.words(speech)
    print(len(speech_words) / len(set(speech_words)))
    print(len(speech_words) / len((speech_sents)))
    

# |%%--%%| <SGR3TBLwk0|U0SKuCNGzl>

speech_len = [(len(inaugural.words(speech)),speech) for speech in speeches ]
speech_len

# |%%--%%| <U0SKuCNGzl|fyFbcfLHQl>

print(max(speech_len))
print(min(speech_len))

# |%%--%%| <fyFbcfLHQl|JD5VbxzSH1>

import pandas as pd

# |%%--%%| <JD5VbxzSH1|O9AwYhFfAx>

data = pd.DataFrame([int(speech[:4]), len(inaugural.words(speech))/len(inaugural.sents(speech))] for speech in speeches)

# |%%--%%| <O9AwYhFfAx|IQh68FVOtu>

data.head()

# |%%--%%| <IQh68FVOtu|roBcSKjJ7a>

data.columns = ['Year', 'Average WPS']

# |%%--%%| <roBcSKjJ7a|oqX26rpcqd>

import matplotlib.pyplot as plt
data.plot('Year')
