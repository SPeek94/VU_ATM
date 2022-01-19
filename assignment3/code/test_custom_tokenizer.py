import spacy
import pandas as pd
import numpy as np
import re
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.tokens import Token

nlp = spacy.load("en_core_web_sm")

nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r"(?<=Chapter )\d+.").match)


text = "Chapter 1. Mr. Sherlock Holmes"

regex = r"(Chapter \d+)(\.)"

re.sub(regex, r"\1", text)


doc = nlp(text)


for token in doc:
    print(token.text)
