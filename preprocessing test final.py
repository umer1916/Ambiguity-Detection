import os
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import download

download('punkt')
download('stopwords')
download('wordnet')

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
   
    sentences = sent_tokenize(text)
    tokens = [nlp(sentence) for sentence in sentences]

    noun_phrases = []
    for doc in tokens:
        noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
        if noun_count >= 2:
            noun_phrases.extend([token.text for token in doc if token.pos_ == 'NOUN'])

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [[lemmatizer.lemmatize(token.text) for token in doc] for doc in tokens]

    porter = PorterStemmer()
    stemmed_tokens = [[porter.stem(word) for word in sentence] for sentence in lemmatized_tokens]

    suffixes = ('s', 'es', 'ed', 'ing')
    suffix_stripped_tokens = [[word[:-1] if word.endswith(suffixes) else word for word in sentence] for sentence in stemmed_tokens]

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [[word for word in sentence if word.lower() not in stop_words] for sentence in suffix_stripped_tokens]

    return noun_phrases, filtered_tokens, tokens, stemmed_tokens, lemmatized_tokens

def full_string_matching(np1, np2):
    return np1 == np2

def headword_matching(np1, np2):
    return np1.split()[0] == np2.split()[0]

def modifier_matching(np1, np2):
    return set(np1.split()) == set(np2.split())

def is_definite_or_demonstrative_np(np):
    doc = nlp(np)
    for token in doc:
        if token.dep_ in ['det', 'poss'] and token.text.lower() in ['the', 'this', 'that', 'these', 'those']:
            return True
    return False

def is_proper_name(np):
    doc = nlp(np)
    for token in doc:
        if token.pos_ == 'PROPN':
            return True
    return False

def number_agreement(np1, np2):
    doc1 = nlp(np1)
    doc2 = nlp(np2)
    return all(token1.tag_ == token2.tag_ for token1, token2 in zip(doc1, doc2))

def pp_attachment(np1, np2):
    doc1 = nlp(np1)
    doc2 = nlp(np2)
    for token1 in doc1:
        for child in token1.children:
            if child.text in doc2.text:
                return True
    return False

def appositive(np1, np2):
    doc1 = nlp(np1)
    doc2 = nlp(np2)
    for token1 in doc1:
        if token1.dep_ == 'appos' and token1.head.text == np2:
            return True
    for token2 in doc2:
        if token2.dep_ == 'appos' and token2.head.text == np1:
            return True
    return False

def syntactic_role(np1, np2):
    doc1 = nlp(np1)
    doc2 = nlp(np2)
    roles1 = [token.dep_ for token in doc1]
    roles2 = [token.dep_ for token in doc2]
    return roles1 == roles2

def semantic_class(np1, np2):
    return np1.split()[0] == np2.split()[0]

if __name__ == "__main__":
  
    dataset_path = "C:\\Users\\Dell\\Desktop\\FYP\\test"

    file_path = os.path.join(dataset_path, "test.xlsx")
    df = pd.read_excel(file_path)

    for index, row in df.iterrows():
        text = row['Context(cj)']

        noun_phrases, filtered_tokens, tokens, stemmed_tokens, lemmatized_tokens = preprocess_text(text)

        print("Original Sentence:")
        print(text)
        print()

        if noun_phrases:
            print("The sentence has at least two nouns in the antecedent.")
            print("Noun Phrases:", noun_phrases)
        else:
            print("The sentence doesn't have at least two nouns in the antecedent.")

        print("Filtered Tokens after Stop Words Removal:")
        print(filtered_tokens)
        print()

        for i in range(len(noun_phrases)):
            for j in range(i+1, len(noun_phrases)):
                np1 = noun_phrases[i]
                np2 = noun_phrases[j]
                print(f"NP{i+1}:", np1)
                print(f"NP{j+1}:", np2)
                print("Full String Matching:", 'Y' if full_string_matching(np1, np2) else 'N')
                print("Headword Matching:", 'Y' if headword_matching(np1, np2) else 'N')
                print("Modifier Matching:", 'Y' if modifier_matching(np1, np2) else 'N')
                print("NP Type (NPi):", 'Y' if is_definite_or_demonstrative_np(np1) else 'N')
                print("NP Type (NPj):", 'Y' if is_definite_or_demonstrative_np(np2) else 'N')
                print("Proper Name:", 'Y' if is_proper_name(np1) and is_proper_name(np2) else 'N')
                print("Number Agreement:", 'Y' if number_agreement(np1, np2) else 'N')
                print("PP Attachment:", 'Y' if pp_attachment(np1, np2) else 'N')
                print("Appositive:", 'Y' if appositive(np1, np2) else 'N')
                print("Syntactic Role:", 'Y' if syntactic_role(np1, np2) else 'N')
                print("Semantic Class:", 'Y' if semantic_class(np1, np2) else 'N')
                print()

        print()
