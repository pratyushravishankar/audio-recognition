import utils
from sklearn.model_selection import train_test_split
import lsh_random_projection as LSH
import pickle


def build_model():

    features = utils.load("data/fma_metadata/features.csv")
    tracks = utils.load('data/fma_metadata/tracks.csv')

    non_nulls_tracks = tracks[tracks['track']['genre_top'].notnull()]

    print(non_nulls_tracks.head())

    non_null_features = features.loc[non_nulls_tracks.index]

    X_train, X_test = train_test_split(non_null_features, test_size=1)
    lsh = LSH.LSH(1, 15, 140)
    lsh.add(X_train['mfcc'])

    save_object(lsh, 'lsh.pkl')


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


build_model()

with open('lsh.pkl', 'rb') as input:
    lsh = pickle.load(input)
    # lsh.get()
