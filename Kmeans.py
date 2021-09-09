from numpy import unique
from sklearn.cluster import KMeans
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification


X, _ = make_classification(n_samples=7500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

Km = KMeans(n_clusters=50)

Km.fit(X)

K = Km.predict(X)

clusters = unique(K)

for c in clusters:
	row_ix = where(K == c)
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])

pyplot.show()
