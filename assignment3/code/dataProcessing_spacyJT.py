import spacy
import pandas as pd
import numpy as np
import re
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en_core_web_sm")
print("imports complete")

def LoadAnnotatedData(inputfile):
    """"""

    df_unprocessed = pd.read_csv(inputfile, header=None, encoding="utf-8",
        sep="\t",
        index_col=False,
    )

    # Somehow this fixes the `` issues
    nlp.tokenizer = Tokenizer(nlp.vocab)
    df_unprocessed.columns = ["Chapter", "Sentence_ID", "Token_ID", "Token", "Negation_cue"]
    df_unprocessed["Sentence_ID_unique"] = df_unprocessed.groupby(["Chapter", "Sentence_ID"]).ngroup()
    regex = r"(Chapter \d+)(\.)"
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("``", "\"")
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("''", "\"")


    token_sent = []
    for i in range(len(set(df_unprocessed["Sentence_ID_unique"]))):
        if i%500 == 0:
            print('loop', i, "/3700?")
        sent = " ".join(df_unprocessed[df_unprocessed.Sentence_ID_unique == i]["Token"])
        cue = list(df_unprocessed[df_unprocessed.Sentence_ID_unique == i]["Negation_cue"])
        fixed_sent = re.sub(regex, r"\1", sent)
        doc = nlp(fixed_sent)

        token_sent.append((doc, cue))

    return token_sent

def create_parsed_df(token_sentences):
    """"""

    listOfDicts = []
    sent_idx = 0
    for doc, cues in token_sentences:  # for each tokenized/processed sentence in the list
        for (
            sent
        ) in doc.sents:  # take each sentence, since we only have 1 sentence it loops 1 time
            for i, word in enumerate(sent):  # go trough each token/word

                if word.head == word:  # reset counter
                    head_idx = 0
                else:  # otherwise calculate idx
                    head_idx = word.head.i - sent[0].i + 1
                dict_parser_output = (
                    {}
                )  # make dictionary and fill it values, as showed in the report appendix II
                dict_parser_output["idx_sent"] = sent_idx
                dict_parser_output["Token_ID"] = i
                dict_parser_output["Token"] = word.text
                dict_parser_output["Lemma"] = word.lemma_
                dict_parser_output["POS"] = word.pos_
                dict_parser_output["POS_TAG"] = word.tag_
                dict_parser_output["Dependency_Head"] = head_idx
                dict_parser_output["Dependency_Label"] = word.dep_
                dict_parser_output["Negation_cue"] = cues[i]


                listOfDicts.append(dict_parser_output)  # append to list
        sent_idx += 1

    columns_ = [
        "Token_ID",
        "Token",
        "Lemma",
        "POS",
        "POS_TAG",
        "Dependency_Head",
        "Dependency_Label",
        "idx_sent",
        "Negation_cue",
    ]


    df_output_parser = pd.DataFrame(listOfDicts, columns=columns_)
    df_output_parser["next"] = df_output_parser.Token.shift(fill_value="None")
    df_output_parser["prev"] = df_output_parser.Token.shift(-1, fill_value="None")


    return df_output_parser

def CreateHighExpBools(dataframe, wordcolumn, POScolumn, FrequentSet = ("n't", 'never', 'no', 'none', 'nor', 'not', 'nothing', 'without')):
    PosVocab = []
    for i, word in enumerate(dataframe[wordcolumn]):
        if dataframe[POScolumn][i] in ['NOUN', 'ADV', 'VERB', 'ADJ', 'VBN']:
            PosVocab.append(word)

    HighExpCueBool = []
    HighExpAffixBool = []
    for word in dataframe[wordcolumn]:
        word = word.lower()
        if word[:2] in ['im', 'ir', 'no', 'un', 'in'] and word[2:] in PosVocab:
            HighExpAffixBool.append(1)
        elif word[:3] in ['dis'] and word[3:] in PosVocab:
            HighExpAffixBool.append(1)
        elif word.endswith(('less', 'lessly', 'lessness')) and word not in ['unless','bless']: #and word[-4:] in Vocab:
            HighExpAffixBool.append(1)
        else:
            HighExpAffixBool.append(0)

        if word in FrequentSet:
            HighExpCueBool.append(1)
        else:
            HighExpCueBool.append(0)

    print(sum(HighExpCueBool))

    dataframe['HECT'] = HighExpCueBool
    dataframe['HECA'] = HighExpAffixBool

    return dataframe

# feat = CreateHighExpBool(parseddf, "Token", "POS")

def main(unprocessed_file, wordcolumn = "Token", POScolumn = 'POS'):
    print("Loading File into SpaCy")
    loadedsents = LoadAnnotatedData(unprocessed_file)
    print("Building Features DataFrame")
    parsed_dataframe = create_parsed_df(loadedsents)
    print("Adding Expectancy Bool")
    Features_df = CreateHighExpBools(parsed_dataframe, wordcolumn = "Token", POScolumn = 'POS')

    outfile = unprocessed_file.replace(".txt", ".Preprocessed.csv")
    Features_df.to_csv(outfile)
    print("PreProcessed File Created")

    return Features_df

main('../Data/tmdev.txt')




# df_output_parser["Negation_cue"] = df_train["Negation_cue"]

#print(df_output_parser.head(6))  # shows results partly

# keys = ["Token", "Token_ID", "Negation_cue"]


# df_train_check = df_train[keys]
# df_output_parser_check = df_output_parser[keys]

# df_train_check.compare(df_output_parser_check)


# def check_correct_negCue_labels(df_gold, df_parsed):
#     pass

