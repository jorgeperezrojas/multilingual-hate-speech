# requiere:
# - keras 
# - gensim
# - spacy (para data 'es', 'en', 'it', hacer "python -m spacy download es" y similar para 'en' y 'it')

import raw_data
import numpy as np
from to_tokens import Tokenizer
from word_vectors import WordVectors
from hs_models import LSTM_model
import utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from multiprocessing import Pool

RANDOM_SEED = 33
NUM_PROCESS = 5
max_len = 40
vector_size = 300
debug = True

languages=['en','es','it']
emb_files = ['wiki.en.100k.align.vec','wiki.es.100k.align.vec','wiki.it.100k.align.vec']
#emb_files = ['wiki.500k.en.align.vec','wiki.500k.es.align.vec','wiki.500k.it.align.vec']
base_emb_path = '/Users/jperez/research/nlp/word-embeddings/multilingual/'
#base_emb_path = '/data/word_embeddings/multilingual/fasttext/small/'

labels=['en','es','it','manual']
labels_language={'en':'en','es':'es','it':'it','manual':'es'}

tokenizer_options = {
    'delete_punctuation':True, 
    'delete_urls':True, 
    'delete_mentions':True, 
    'delete_hashtags':True,
}
tokenizer = Tokenizer(languages=languages)
data_tokens_x = {}
for label in raw_data.data_x:
    data_x = raw_data.data_x[label]
    language = labels_language[label]
    data_tokens_x[label] = tokenizer.tokenize_list_of_texts(data_x, language, **tokenizer_options)

wv = WordVectors(languages, base_emb_path, emb_files, vector_size)

# Lo siguiente crea arrays con las secuencias de vectores para todas las frases del corpus.
# Con los datos actuales ocupa +2GB de RAM.
# Deberíamos reemplazarlo por un generador (sobre todo si los datos crecen).
data_array_x = {}
for label in data_tokens_x:
    language = labels_language[label]
    data_array_x[label] = wv.vec_seq_batch_array(data_tokens_x[label], language, max_len)

data_array_y = {}
for label in raw_data.data_y:
    data_array_y[label] = np.array(raw_data.data_y[label]).reshape(-1,1)

split_size = 0.25

# este es el loop principal de entrenamiento
def train_scenario(scenario, epochs=30, verbose=2, debug=debug):
    x_train, x_test, y_train, y_test = utils.split_data_from_description(
        data_array_x, data_array_y, scenario, split_size, RANDOM_SEED)

    if debug:
        epochs = 1
        verbose = 2
        x_train = x_train[:50]
        y_train = y_train[:50]
        x_test = x_test[:10]
        y_test = y_test[:10]

    hyper_params_choices = [
        {'batch_size':16},
        {'batch_size':32},
        {'batch_size':64},
    ]

    for hparams in hyper_params_choices:

        model = LSTM_model(max_len)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        hashstr = utils.hash_texto(str(scenario) + str(hparams))[:15]
        filename =  'model_' + hashstr + '.h5'
        model_checkpoint = ModelCheckpoint(filename, monitor='val_acc', save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        print('starting training',hashstr)

        h = model.fit(x_train, y_train, epochs=epochs, batch_size=hparams['batch_size'], 
            validation_data=(x_test, y_test), verbose=verbose,
            shuffle=True, callbacks=[early_stop, model_checkpoint])

        model = load_model(filename)
        (loss, acc) = model.evaluate(x_test,y_test)
        with open('results.txt', 'a+') as results_file, open('details.txt', 'a+') as details_file:
            info_results = 'scenario:' + str(scenario)
            info_results += '\tacc:' + str(acc)[:6] + '\tloss:' + str(loss)[:6] 
            info_results += '\tmodel:' + filename + '\n'
            results_file.write(info_results)

            info_details = 'scenario:' + str(scenario) 
            info_details += '\tmodel:' + filename + '\thiperpars:' + str(hparams) + '\thisotry:' + str(h.history) + '\n'
            details_file.write(info_details)
        
        print('finishing training',hashstr)
        print('scenario:',scenario,'best acc:',acc,'hiper params',str(hparams))

scenarios = [
    (['en'], ''),
    (['es'], ''),
    (['it'], ''),
    (['en','es'], ''),
]

# corre procesos simultaneos para el entrenamiento
p = Pool(NUM_PROCESS)
p.map(train_scenario, scenarios)
    

