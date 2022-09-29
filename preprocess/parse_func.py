from turtle import title
from icecream import ic
import pandas as pd
import json
from nltk.tokenize import word_tokenize

def parsing_task(task):
    text = word_tokenize(task)
    sentence = nltk.pos_tag(text)
    NP_grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(NP_grammar)
    result1 = cp.parse(sentence)

    VP_grammar = "VP: {<VB><TO>*<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(VP_grammar)
    result2 = cp.parse(sentence)
    return result1, result2