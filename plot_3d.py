import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#  load and process the raw data
try:
    data = pd.read_csv("creditcards.csv")
except:
    print("Can't find 'creditcards.csv'")
    quit()

#  attempt to load the custom labels
loaded = False
try:
    labels = pd.read_csv("labels.csv")
    loaded = True
except:
    print("Can't find the custom labels, graphing without them")

#  extract top three principal components
data_pca = pd.DataFrame(PCA(3).fit_transform(data))
data_pca.columns = ['PC1', 'PC2', 'PC3']

#  graph the data
plt.style.use('ggplot')
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection = '3d')
if loaded:
    ax.scatter(data_pca['PC1'], data_pca['PC2'], data_pca['PC3'], c=labels)
else:
    ax.scatter(data_pca['PC1'], data_pca['PC2'], data_pca['PC3'])

#  label the principal components
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Graphing along first three principal components")

#  display!
plt.show()
