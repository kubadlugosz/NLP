"""°°°
## Intro to NLTK
°°°"""
# |%%--%%| <DoI1dZEW85|SbMDV3G0Gt>

import nltk 
#Project Gutenburg - open source database for books and text
nltk.download('gutenberg') #download gutenburg

# |%%--%%| <SbMDV3G0Gt|gaH9VeFzyr>

nltk.corpus.gutenberg.fileids()

# |%%--%%| <gaH9VeFzyr|CtP4Y6YDPJ>

#Load the book moby dick
md = nltk.corpus.gutenberg.words('melville-moby_dick.txt')

# |%%--%%| <CtP4Y6YDPJ|ry24BeUcgJ>

md

# |%%--%%| <ry24BeUcgJ|EKohwHuX6i>

md_set =set(md)

# |%%--%%| <EKohwHuX6i|q70WXSlkrL>

len(md_set)

# |%%--%%| <q70WXSlkrL|W15ZKXCBCN>

len(md) / len(md_set)

# |%%--%%| <W15ZKXCBCN|m4VHOds1OR>

md_sents = nltk.corpus.gutenberg.sents('melville-moby_dick.txt')

# |%%--%%| <m4VHOds1OR|qc8RLbukjU>

len(md) / len(md_sents)

# |%%--%%| <qc8RLbukjU|lOK9rht6KF>
"""°°°
## Example-Words Per Sentence Trends
°°°"""
# |%%--%%| <lOK9rht6KF|CCCN1nptgT>

from nltk.corpus import inaugural
nltk.download('inaugural')

# |%%--%%| <CCCN1nptgT|r3qEjyM5X1>

inaugural.fileids()

# |%%--%%| <r3qEjyM5X1|MaAbqX790P>

speeches = inaugural.fileids()
for speech in speeches:
    speech_sents = inaugural.sents(speech)
    speech_words = inaugural.words(speech)
    print(len(speech_words) / len(set(speech_words)))
    print(len(speech_words) / len((speech_sents)))
    

# |%%--%%| <MaAbqX790P|CgphqiTRiV>

speech_len = [(len(inaugural.words(speech)),speech) for speech in speeches ]
speech_len

# |%%--%%| <CgphqiTRiV|Z3aswQJu6y>

print(max(speech_len))
print(min(speech_len))

# |%%--%%| <Z3aswQJu6y|kR5P4YEjjO>

import pandas as pd

# |%%--%%| <kR5P4YEjjO|yBCKQVwWuC>

data = pd.DataFrame([int(speech[:4]), len(inaugural.words(speech))/len(inaugural.sents(speech))] for speech in speeches)

# |%%--%%| <yBCKQVwWuC|rtkcwZVCyD>

data.head()

# |%%--%%| <rtkcwZVCyD|8fyy0whnjI>

data.columns = ['Year', 'Average WPS']

# |%%--%%| <8fyy0whnjI|OO8JzcmzR3>

import matplotlib.pyplot as plt
data.plot('Year')
