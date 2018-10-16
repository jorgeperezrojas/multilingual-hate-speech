from gensim.models.keyedvectors import KeyedVectors
import numpy as np

class WordVectors():
    
    def __init__(self, 
            languages=['en','es','it'],
            base_path='/Users/jperez/research/nlp/word-embeddings/multilingual/',
            filenames=['wiki.en.100k.align.vec','wiki.es.100k.align.vec','wiki.it.100k.align.vec'],
            vector_size=300,
            verbose=True
            ):

        self.languages = languages
        self.vectors = {}
        self.vector_size = vector_size
        for i,l in enumerate(languages):
            fname = base_path + filenames[i]
            if verbose:
                print('Loading ' + l + ' vectors from ' + fname + ' ...')
            self.vectors[l] = KeyedVectors.load_word2vec_format(fname, binary=False)

    def avg_vec(self, tokens, language):
        size = self.vector_size
        avg_vec = np.zeros((size,), dtype="float32")
        n_words = 0

        if language not in self.languages:
            print('Warning: Language ' + language + ' not in loaded languages ' + str(self.languages))
            return avg_vec
        
        for word in tokens:
            if word in self.vectors[language]:
                n_words += 1
                avg_vec = self.vectors[language][word]
        if n_words > 0:
            avg_vec = (1 / n_words) * avg_vec
        else: # if there is no info in the sentences just output a random vector
            avg_vec = np.random.rand(size)
        return avg_vec

    # TODO: iterar desde el final de la secuencia para asegurar que el masking funcione correctamente
    def vec_seq_array(self, sequence, language, max_len=40):
        out_array = np.zeros((max_len,self.vector_size), dtype="float32")
        tokens = sequence[-max_len:]
        i = max_len - len(tokens)
        for token in tokens:
            if token in self.vectors[language]:
                out_array[i,:] = self.vectors[language][token]
            i = i + 1
        return out_array

    def vec_seq_batch_array(self, sequences, language, max_len=40):
        list_of_arrays = []
        for seq in sequences:
            list_of_arrays.append(self.vec_seq_array(seq, language, max_len))
        out_array = np.stack(list_of_arrays, axis=0)
        return out_array


