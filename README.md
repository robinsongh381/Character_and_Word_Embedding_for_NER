# Named Entity Recognition
Implementation of NER with character and word level embeddings
For each word, its word embedding, character embedding and POS-tag embedding are concatenated, which is then fed into bidirectional LSTM. Note that CNN is used to embed characters whose kernel sizes and number of channels are defined in `./utils/constants.py`.

## Language & Tokenizer
This implementation is based on **Korean** and hence *Okt* tokenizer is used from `konlpy.tag`.

## Data
The data is provided from [here](https://github.com/kmounlp/NER).

## Process
##### 1. Build Vocab and Preprocess
run vocab.py   
1. build vocabs for wordss, characters and pos-taggings
2. tokenize and convert into index for both tokens and labels
3. 
##### 2. Train
run train.py

##### 3. Inference
run inference.ipynb
Make sure that you give a relevant path for your trained model