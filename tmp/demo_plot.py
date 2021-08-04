import seaborn as sns
import matplotlib.pyplot as plt
import ssl
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

tips = sns.load_dataset("tips")
print(tips.head())

sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
plt.show()

df = pd.read_csv('2016.csv')
print(df.info())
print(df.describe())
print(df.head())
grid = sns.FacetGrid(df, col="Region", hue="Region", col_wrap=5)
grid.map(sns.scatterplot, "Economy (GDP per Capita)", "Health (Life Expectancy)")

grid.add_legend()

plt.show()

###### 3D
sns.set(style="darkgrid")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = df['Happiness Score']
y = df['Economy (GDP per Capita)']
z = df['Health (Life Expectancy)']

ax.set_xlabel("Happiness")
ax.set_ylabel("Economy")
ax.set_zlabel("Health")

ax.scatter(x, y, z)

plt.show()

##### 3d
import re, seaborn as sns
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# generate data
n = 200
x = np.random.uniform(1, 20, size=n)
y = np.random.uniform(1, 100, size=n)
z = np.random.uniform(1, 100, size=n)

# axes instance
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

# plot
sc = ax.scatter(x, y, z, s=40, c=x, marker='o', cmap=cmap, alpha=1)
# sc = ax.scatter(x, y, z, s=40, c=x, marker='o', alpha=1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
# plt.grid(b=True)
# plt.tight_layout(pad = 2.0, rect=(0.3, 0.3, 0.8, 0.8))
# save
plt.savefig("scatter_hue", bbox_inches='tight')
plt.show()
