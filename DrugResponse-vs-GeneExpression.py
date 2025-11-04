#Drug response versus gene expression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = {
    'Patient_ID': [f'P{i}' for i in range(1, 16)],
    'GeneA_expression': [2.1, 2.5, 3.0, 1.8, 2.9, 3.5, 2.7, 3.3, 1.9, 2.8, 3.1, 2.6, 3.4, 2.0, 2.3],
    'GeneB_expression': [5.1, 5.3, 4.8, 6.2, 5.0, 4.5, 5.6, 4.9, 6.4, 5.2, 4.7, 5.4, 4.6, 6.1, 5.8],
    'Drug_Response':    [0.65, 0.72, 0.85, 0.58, 0.81, 0.90, 0.75, 0.83, 0.55, 0.78, 0.82, 0.70, 0.88, 0.62, 0.68]
}
df = pd.DataFrame(data)
print(df)
print(df.corr(numeric_only=True))

plt.scatter(df["GeneA_expression"],df["Drug_Response"],marker='o')
plt.title("GeneA Expression vs Drug Response")
plt.xlabel("GeneA Expression")
plt.ylabel("Drug Response")
plt.grid(color = 'blue', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.show()

sns.heatmap(df.corr(numeric_only=True),annot= True, cmap= "magma")
plt.title ("Correlation matrix: Genes vs Drug Response")
plt.show()



