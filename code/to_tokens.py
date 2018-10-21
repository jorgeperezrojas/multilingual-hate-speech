import spacy
import sys
import re

class Tokenizer:

    def __init__(self, languages=['en','es','it']):
        self.languages=languages
        self.language_models = {}
        for l in languages:
            self.language_models[l] = spacy.load(l)

    # TODO: decidir si sacar emojis (usar spacymoji https://github.com/ines/spacymoji)
    def tokenize_text(self, text, language, 
            delete_punctuation=True, 
            delete_urls=True, 
            delete_mentions=True, 
            delete_hashtags=True
        ):
    
        doc = self.language_models[language](text)
        tokens = []
        for token in doc:
            if token.is_space:
                continue
            if delete_hashtags and token.i > 0 and doc[token.i - 1].text == '#':
                continue
            if delete_punctuation and (token.is_punct or token.text == '\n'):
                continue
            if delete_urls and token.like_url:
                continue
            if delete_mentions and token.text.startswith('@'):
                continue
            clean_text = token.text.lower()
            clean_text = re.sub('(\\n|\\t)+','',clean_text)
            if clean_text.strip() == '':
                continue
            tokens.append(clean_text)
        return tokens   

    def tokenize_list_of_texts(self, list_of_texts, language, verbose=True, **kwargs):
        out_list = []
        if verbose:
            print('tokenizing list of',len(list_of_texts),'elements, language:',language)
        for i,text in enumerate(list_of_texts):
            if verbose:
                sys.stdout.write('\r' + str(i+1) + '/' + str(len(list_of_texts)))
            tokens = self.tokenize_text(text, language, **kwargs)
            out_list.append(tokens)
        if verbose:
            print('\ndone!')
        return out_list
