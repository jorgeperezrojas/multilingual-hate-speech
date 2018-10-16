import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

data_x = {}
data_x['es'] = load_object('textos_esp.pkl')
data_x['en'] = load_object('textos_ing.pkl')
data_x['it'] = load_object('textos_ita.pkl')
data_x['manual'] = load_object('textos_manual.pkl')

data_y = {}
data_y['es'] = load_object('textos_esp_class.pkl')
data_y['en'] = load_object('textos_ing_class.pkl')
data_y['it'] = load_object('textos_ita_class.pkl')
data_y['manual'] = load_object('manual_class.pkl')


