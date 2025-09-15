import pandas as pd
import matplotlib.pyplot as plt

# Load your data
d = pd.read_csv('df_tmid1_labeled.csv')

# Create scatter plot with color based on cluster label
plt.scatter(d['umap1'], d['umap2'], c=d['Cluster_Label'], cmap='tab10', s=10)
plt.colorbar(label='Cluster Label')  # Optional: shows a color legend
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP Clustering Visualization')
plt.show()
