import urllib.request #Handling URL
from bs4 import BeautifulSoup  #HAndle parsing html files
import nltk
import matplotlib.pyplot as plt


nltk.download( 'stopwords') #is was extract
from nltk.corpus import stopwords


response = urllib.request.urlopen('https://en.wikipedia.org/wiki/Mahendra')
html = response.read()
#it is difficult to read html file thats why we use parsing
#using html5lib
soup = BeautifulSoup(html, 'html5lib')
text = soup.get_text(strip = True)


tokens = [ t for t in text.split()]

sr = stopwords.words('english')
clean_tokens = tokens[:]

for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)

freq = nltk.FreqDist(clean_tokens)
for key, val in freq.items():
    print(str(key) + ':'+ str(val))

freq.plot(20, cumulative = False)
print(freq.most_common(20))
