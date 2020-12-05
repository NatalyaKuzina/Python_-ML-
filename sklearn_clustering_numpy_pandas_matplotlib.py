from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
df = pd.read_csv('blobs.csv')
X = df.iloc[:, 0:]
clustering = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
          damping=0.5, max_iter=1, preference=-0.14, verbose=False).fit(X)
clustering
clustering.labels_

plt.scatter(df.X, df.Y, c = clustering.labels_, cmap= 'rainbow' )
plt.show()

clustering = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
          damping=0.5, max_iter=1, preference=-0.15, verbose=False).fit(X)
clustering
clustering.labels_

plt.scatter(df.X, df.Y, c = clustering.labels_, cmap= 'rainbow' )
plt.show()

clustering = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
          damping=0.5, max_iter=1, preference=-0.12, verbose=False).fit(X)
clustering
clustering.labels_

plt.scatter(df.X, df.Y, c = clustering.labels_, cmap= 'rainbow' )
plt.show()
