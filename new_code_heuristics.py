import os
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spacy.matcher import Matcher
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Tokenization
    sentences = sent_tokenize(text)
    tokens = [nlp(sentence) for sentence in sentences]

    # Identify noun phrases and pronouns
    noun_phrases = []
    pronouns = []
    for doc in tokens:
        np_count = 0  # Counter for noun phrases in the current sentence
        for token in doc:
            if token.pos_ == 'NOUN':
                noun_phrases.append(token.text)
                np_count += 1
            elif token.pos_ == 'PRON' and np_count >= 2:
                pronouns.append(token.text)
        if np_count >= 2:
            break  # Stop processing further sentences in this document if we found two noun phrases
 # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [[lemmatizer.lemmatize(token.text) for token in doc] for doc in tokens]

    # Porter Stemming
    porter = PorterStemmer()
    stemmed_tokens = [[porter.stem(word) for word in sentence] for sentence in lemmatized_tokens]

    # Suffix Stripping
    suffix_stripped_tokens = [[word[:-1] if word.endswith(('s', 'es', 'ed', 'ing')) else word for word in sentence] for sentence in stemmed_tokens]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [[word for word in sentence if word.lower() not in stop_words] for sentence in suffix_stripped_tokens]

    return noun_phrases, pronouns
# Coreference Heuristics
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

def is_proper_name_np(np):
    doc = nlp(np)
    for token in doc:
        if token.pos_ == 'PROPN':
            return True
    return False

def number_agreement_np(np1, np2):
    doc1 = nlp(np1)
    doc2 = nlp(np2)
    for token1, token2 in zip(doc1, doc2):
        if token1.tag_ != token2.tag_:
            return False
    return True

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

# Antecedent Heuristics
def extended_number_agreement(np, pron):
    if number_agreement(np, pron):
        return 'Y'
    elif is_proper_name(pron):
        return 'N_P'
    else:
        return 'N'

def number_agreement(np, pron):
    doc_np = nlp(np)
    doc_pron = nlp(pron)
    return (any(token.tag_ in ('NN', 'NNP') for token in doc_np) and any(token.tag_ in ('PRP') for token in doc_pron)) or \
           (any(token.tag_ in ('NNS', 'NNPS') for token in doc_np) and any(token.tag_ in ('PRP$') for token in doc_pron))

def is_proper_name(pron):
    doc = nlp(pron)
    return any(token.pos_ == 'PROPN' for token in doc)

def definiteness(pron):
    doc = nlp(pron)
    for token in doc:
        if token.dep_ == 'det' and token.text.lower() in ['the', 'this', 'that', 'these', 'those']:
            return 'Y'
        if token.dep_ == 'poss':
            return 'Y'
    return 'N'

def non_prepositional(pron):
    doc = nlp(pron)
    for token in doc:
        if token.dep_ == 'prep':
            return 'N'
    return 'Y'

def syntactic_constraint(np, pron):
    doc_np = nlp(np)
    doc_pron = nlp(pron)
    roles_np = {token.dep_ for token in doc_np}
    roles_pron = {token.dep_ for token in doc_pron}
    return 'Y' if roles_np & roles_pron else 'N'

def syntactic_parallelism(np, pron):
    return syntactic_constraint(np, pron)

def coordination_pattern(np, pron):
    doc_np = nlp(np)
    doc_pron = nlp(pron)
    for token in doc_np:
        if token.dep_ == 'conj' and token.text in pron:
            return 'Y'
    return 'N'

def non_associated(pron):
    doc = nlp(pron)
    for token in doc:
        if token.dep_ == 'appos':
            return 'N'
    return 'Y'

def indicating_verb(pron, verbs):
    doc = nlp(pron)
    for token in doc:
        if token.lemma_ in verbs:
            return 'Y'
    return 'N'

def semantic_constraint(np, pron):
    synsets_np = wn.synsets(np)
    synsets_pron = wn.synsets(pron)
    if not synsets_np or not synsets_pron:
        return 'N'
    return 'Y' if synsets_np[0].lowest_common_hypernyms(synsets_pron[0]) else 'N'

def semantic_parallelism(np, pron):
    return semantic_constraint(np, pron)

def domain_specific_term(pron, domain_terms):
    return 'Y' if any(term in pron for term in domain_terms) else 'N'

def centering(pron, context):
    return 'Y' if context.count(pron) > 1 else 'N'

def section_heading(pron, heading):
    return 'Y' if pron in heading else 'N'

def sentence_recency(pron, sentence):
    return 'INTRA_S' if pron in sentence else 'INTER_S'

def proximal(pron, sentence):
    np_list = [token.text for token in nlp(sentence) if token.pos_ == 'NOUN']
    if pron in np_list:
        return np_list.index(pron) + 1
    return 'N/A'

def local_collocation_frequency(pron, context):
    return context.count(pron)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xlsx'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def bnc_collocation_frequency(pron, word_list):
    return 'Y' if pron in word_list else 'N'
    # Example usage:
if __name__ == "__main__":
    # Load the Excel file
    file_path = "ReqEval.xlsx"
    df = pd.read_excel(file_path)

    # Sample indicating verbs and domain-specific terms
    indicating_verbs = ['mention', 'state', 'describe']
    domain_terms = ['requirement', 'specification', 'document']

    # Initialize an empty list to store the results
    results = []

    processed_sentences = set()

    # Iterate through each row in the dataframe
    for index, row in df.iterrows():
        # Extract the text from the dataframe
        text = row['Context(cj)']

        # Skip processing if the sentence has already been processed
        if text in processed_sentences:
            continue
        processed_sentences.add(text)

        # Apply preprocessing techniques
        noun_phrases, pronouns = preprocess_text(text)

        # Check for the required conditions
        if len(noun_phrases) < 2 or not pronouns:
            continue

        np1 = noun_phrases[0]
        np2 = noun_phrases[1]
        pron = pronouns[0]

        # Check heuristics as required
        # Example heuristics (replace with your actual heuristics)
        all_heuristics_checked = True
        if not full_string_matching(np1, np2):
            all_heuristics_checked = False
        if not headword_matching(np1, np2):
            all_heuristics_checked = False
        if not modifier_matching(np1, np2):
            all_heuristics_checked = False
        if not is_definite_or_demonstrative_np(np1):
            all_heuristics_checked = False
        if not is_proper_name_np(np1):
            all_heuristics_checked = False
        if not number_agreement_np(np1, np2):
            all_heuristics_checked = False
        if not pp_attachment(np1, np2):
            all_heuristics_checked = False
        if not appositive(np1, np2):
            all_heuristics_checked = False
        if not syntactic_role(np1, np2):
            all_heuristics_checked = False
        if not semantic_class(np1, np2):
            all_heuristics_checked = False

        # Collect results
        for pronoun in pronouns:
            for noun_phrase in noun_phrases:
                result = {
                    "Sentence": text,
                    "Pronoun": pronoun,
                    "Noun Phrase": noun_phrase,
                    "Number Agreement": extended_number_agreement(noun_phrase, pronoun),
                    "Definiteness": definiteness(pronoun),
                    "Non-prepositional": non_prepositional(pronoun),
                    "Syntactic Constraint": syntactic_constraint(noun_phrase, pronoun),
                    "Syntactic Parallelism": syntactic_parallelism(noun_phrase, pronoun),
                    "Coordination Pattern": coordination_pattern(noun_phrase, pronoun),
                    "Non-associated": non_associated(pronoun),
                    "Indicating Verb": indicating_verb(pronoun, indicating_verbs),
                    "Semantic Constraint": semantic_constraint(noun_phrase, pronoun),
                    "Semantic Parallelism": semantic_parallelism(noun_phrase, pronoun),
                    "Domain-specific Term": domain_specific_term(pronoun, domain_terms),
                    "Centering": centering(pronoun, text),
                    "Section Heading": section_heading(pronoun, row.get('Heading', '')),
                    "Sentence Recency": sentence_recency(pronoun, text),
                    "Proximal": proximal(pronoun, text),
                    "Local-based Collocation Frequency": local_collocation_frequency(pronoun, text),
                    "BNC-based Collocation Frequency": bnc_collocation_frequency(pronoun, domain_terms)
                }
                results.append(result)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Save the results to an Excel file
    output_file_path = "processed_results.xlsx"
    results_df.to_excel(output_file_path, index=False)
    
    print(f"Results have been saved to {output_file_path}")