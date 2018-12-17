# word_embeddings
Implementation of several word embeddings

## Current Status
- skipgram model ran on nltk.corpus.brown (1.1million words, 56k vocabulary size)
- Perform rather badly. Measured in terms of finding similar words using cosine similarity. 

## TODO
- Study the evaluation techniques for word embeddings and conduct a properly evaluation.
- Inspect why model did poorly? Possible reason: 1) brown corpus is too small 2) something wrong with the model construction.
- Try to include wikipedia english dumps and train (fastText provides [`get-wikimedia.sh`](https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh) script to download and process wiki dumps) 
Issue: `sh` scripts can't be ran on windows. Wait for main macbook to finish repair and try it out.


## Future
- Implements GloVe, fastText, ELMo
