import matplotlib.pyplot as plt
import seaborn as sns


def DrawScatterPlot(x, y, xLabel, yLabel):
    plt.scatter(x=x, y=y)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()
    return

def DrawHeatMap(matrix):
    plt.subplots(figsize=(20, 9))
    sns.heatmap(matrix, square=False, cbar=True, annot=True, annot_kws={'size': 5})
    plt.show()
    plt.savefig('matrix_heatmap.png')

def DrawPairPlot(matrix):
    sns.set(style='ticks')
    sns.pairplot(matrix, height=2, kind='reg')
    plt.show()
    plt.savefig('pairplot.png')


