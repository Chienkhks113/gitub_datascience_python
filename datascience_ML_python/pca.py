from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



#load MNIST dataset

X, y = datasets.load_digits(return_X_y=True)