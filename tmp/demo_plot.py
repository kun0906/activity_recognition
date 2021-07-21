import seaborn as sns
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

tips = sns.load_dataset("tips")
print(tips.head())

sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip", hue="sex")
plt.show()
