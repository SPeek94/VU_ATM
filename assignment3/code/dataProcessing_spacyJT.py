import spacy
import pandas as pd
import numpy as np
import re
from spacy.tokenizer import Tokenizer


nlp = spacy.load("en_core_web_lg")
print("imports complete")


def LoadAnnotatedData(inputfile):
    """"""

    df_unprocessed = pd.read_csv(
        inputfile,
        header=None,
        encoding="utf-8",
        sep="\t",
        index_col=False,
    )

    # Somehow this fixes the `` issues
    nlp.tokenizer = Tokenizer(nlp.vocab)
    df_unprocessed.columns = [
        "Chapter",
        "Sentence_ID",
        "Token_ID",
        "Token",
        "Negation_cue",
    ]
    df_unprocessed["Sentence_ID_unique"] = df_unprocessed.groupby(
        ["Chapter", "Sentence_ID"]
    ).ngroup()
    regex = r"(Chapter \d+)(\.)"
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("``", '"')
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("`", '"')
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("'", '"')
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("''", '"')

    token_sent = []
    for i in range(len(set(df_unprocessed["Sentence_ID_unique"]))):
        if i % 500 == 0:
            print("loop", i, "/3700?")
        sent = " ".join(df_unprocessed[df_unprocessed.Sentence_ID_unique == i]["Token"])
        cue = list(
            df_unprocessed[df_unprocessed.Sentence_ID_unique == i]["Negation_cue"]
        )
        fixed_sent = re.sub(regex, r"\1", sent)
        doc = nlp(fixed_sent)

        token_sent.append((doc, cue))

    return token_sent


def create_parsed_df(token_sentences):
    """"""

    listOfDicts = []
    sent_idx = 0
    for (
        doc,
        cues,
    ) in token_sentences:  # for each tokenized/processed sentence in the list
        for (
            sent
        ) in (
            doc.sents
        ):  # take each sentence, since we only have 1 sentence it loops 1 time
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
                dict_parser_output["Token_vector"] = word.vector

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
        "Token_vector",
    ]

    # Create dataframe from the list of dicts
    df_output_parser = pd.DataFrame(listOfDicts, columns=columns_)

    # Columns for the next token and including vector
    df_output_parser["next_token"] = df_output_parser.Token.shift(-1, fill_value="None")
    df_output_parser["next_token_vector"] = df_output_parser.Token_vector.shift(
        -1, fill_value="None"
    )
    # Change "None" --> array of zeros shape (300,)
    df_output_parser["next_token_vector"].iloc[-1] = np.zeros(300)

    # Columns for the previous token and including vector
    df_output_parser["prev_token"] = df_output_parser.Token.shift(fill_value="None")
    df_output_parser["prev_token_vector"] = df_output_parser.Token_vector.shift(
        fill_value="None"
    )
    # Change "None" --> array of zeros shape (300,)
    df_output_parser["prev_token_vector"].iloc[0] = np.zeros(300)

    # Creating string based trigrams
    df_output_parser["trigram"] = (
        df_output_parser.Token.shift()
        + " "
        + df_output_parser.Token
        + " "
        + df_output_parser.Token.shift(-1)
    )
    df_output_parser.loc[0, "trigram"] = df_output_parser.trigram[
        1
    ]  # For first trigram take the second trigram and copy it (because first token doesn't have a prev token)
    df_output_parser.loc[
        len(df_output_parser) - 1, "trigram"
    ] = df_output_parser.trigram[
        len(df_output_parser) - 2
    ]  # the same as for the first token, the last token doesn't have a next token so copy previous trigram

    # Create list of the tokens inside a trigram
    df_output_parser["trigram_list_tokens"] = df_output_parser.apply(
        lambda x: [x.prev_token, x.Token, x.next_token], axis=1
    )

    df_output_parser["trigram_list_tokens"].iloc[0] = df_output_parser[
        "trigram_list_tokens"
    ].iloc[1]
    df_output_parser["trigram_list_tokens"].iloc[-1] = df_output_parser[
        "trigram_list_tokens"
    ].iloc[-2]

    # Create list of the vectors of the corresponding tokens inside a trigram
    df_output_parser["trigram_list_vectors"] = df_output_parser.apply(
        lambda x: [x.prev_token_vector, x.Token_vector, x.next_token_vector], axis=1
    )

    df_output_parser["trigram_list_vectors"].iloc[0] = df_output_parser[
        "trigram_list_vectors"
    ].iloc[1]
    df_output_parser["trigram_list_vectors"].iloc[-1] = df_output_parser[
        "trigram_list_vectors"
    ].iloc[-2]

    # Create string based bigram based on current token and previous token
    df_output_parser["prev_bigram"] = (
        df_output_parser.Token.shift() + " " + df_output_parser.Token
    )
    df_output_parser.loc[0, "prev_bigram"] = df_output_parser.prev_bigram[1]

    # Create list of tokens bigram based on current token and previous token
    df_output_parser["prev_bigram_list_tokens"] = df_output_parser.apply(
        lambda x: [x.prev_token, x.Token], axis=1
    )
    df_output_parser["prev_bigram_list_tokens"].iloc[0] = df_output_parser[
        "prev_bigram_list_tokens"
    ].iloc[1]

    # Create list of vectors of the corresponding tokens bigram based on current token and previous token
    df_output_parser["prev_bigram_list_vectors"] = df_output_parser.apply(
        lambda x: [x.prev_token_vector, x.Token_vector], axis=1
    )
    df_output_parser["prev_bigram_list_vectors"].iloc[0] = df_output_parser[
        "prev_bigram_list_vectors"
    ].iloc[1]

    # Create string based bigram based on current token and next token
    df_output_parser["next_bigram"] = (
        df_output_parser.Token + " " + df_output_parser.Token.shift(-1)
    )
    df_output_parser.loc[
        len(df_output_parser) - 1, "next_bigram"
    ] = df_output_parser.next_bigram[len(df_output_parser) - 2]

    # Create list of tokens bigram based on current token and next token
    df_output_parser["next_bigram_list_tokens"] = df_output_parser.apply(
        lambda x: [x.Token, x.next_token], axis=1
    )
    df_output_parser["next_bigram_list_tokens"].iloc[-1] = df_output_parser[
        "next_bigram_list_tokens"
    ].iloc[-2]

    # Create list of vectors of the corresponding tokens bigram based on current token and next token
    df_output_parser["next_bigram_list_vectors"] = df_output_parser.apply(
        lambda x: [x.Token_vector, x.next_token_vector], axis=1
    )
    df_output_parser["next_bigram_list_vectors"].iloc[-1] = df_output_parser[
        "next_bigram_list_vectors"
    ].iloc[-2]
    return df_output_parser


def CreateHighExpBools(
    dataframe,
    wordcolumn,
    POScolumn,
    FrequentSet=("n't", "never", "no", "none", "nor", "not", "nothing", "without"),
):
    PosVocab = []
    for i, word in enumerate(dataframe[wordcolumn]):
        if dataframe[POScolumn][i] in ["NOUN", "ADV", "VERB", "ADJ", "VBN"]:
            PosVocab.append(word)

    HighExpCueBool = []
    HighExpAffixBool = []
    for word in dataframe[wordcolumn]:
        word = word.lower()
        if word[:2] in ["im", "ir", "no", "un", "in"] and word[2:] in PosVocab:
            HighExpAffixBool.append(1)
        elif word[:3] in ["dis"] and word[3:] in PosVocab:
            HighExpAffixBool.append(1)
        elif word.endswith(("less", "lessly", "lessness")) and word not in [
            "unless",
            "bless",
        ]:  # and word[-4:] in Vocab:
            HighExpAffixBool.append(1)
        else:
            HighExpAffixBool.append(0)

        if word in FrequentSet:
            HighExpCueBool.append(1)
        else:
            HighExpCueBool.append(0)

    print(sum(HighExpCueBool))

    # HECT: Highly expected Cue Token
    dataframe["HECT"] = HighExpCueBool
    # HECA: Higly expected Cue Affix
    dataframe["HECA"] = HighExpAffixBool

    return dataframe


# feat = CreateHighExpBool(parseddf, "Token", "POS")


def main(unprocessed_file, wordcolumn="Token", POScolumn="POS"):
    print("Loading File into SpaCy")
    loadedsents = LoadAnnotatedData(unprocessed_file)
    print("Building Features DataFrame")
    parsed_dataframe = create_parsed_df(loadedsents)
    print("Adding Expectancy Bool")
    Features_df = CreateHighExpBools(
        parsed_dataframe, wordcolumn="Token", POScolumn="POS"
    )

    outfile = unprocessed_file.replace(".txt", ".Preprocessed.pickle")
    Features_df.to_pickle(outfile)
    print("PreProcessed File Created")

    return Features_df


list_of_files = [
    "tmtrain.txt",
    "tmdev.txt",
    "test\\SEM-2012-SharedTask-CD-SCO-test-cardboard.txt",
    "test\\SEM-2012-SharedTask-CD-SCO-test-circle.txt",
]
for file in list_of_files:
    main(
        f"D:\\Studie\\Business Analytics\\Applied Text Mining\\assignment3\\Data\\{file}"
    )
