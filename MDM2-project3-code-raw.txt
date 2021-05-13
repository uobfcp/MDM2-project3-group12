import numpy as np
from nltk.corpus import wordnet as wn
import pandas as pd
import glove
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import itertools
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.patches as mpatches


'''Retrieve synsets and labels'''
primate = wn.synsets('fruit')[0]
hypoPrimates = list([i for i in primate.closure(lambda s:s.hyponyms())])
hypoPrimates.insert(0, primate)
labelHeaders = [i.lemma_names()[0] for i in hypoPrimates]

canine = wn.synsets('vegetable')[0]
hypoCanines = list([i for i in canine.closure(lambda s:s.hyponyms())])
hypoCanines.insert(0, canine)
labelHeaders += [i.lemma_names()[0] for i in hypoCanines]
hyponyms = hypoPrimates + hypoCanines
print(labelHeaders)



M = [] # Initialize empty co-occurence matrix
def co_occurrence(M, ss):
    for i, synset in enumerate(ss):    #loop through all row vectors, taking the synsets (a synset is a word contained with descriptive data) with i an a index
        data = []  #data represents the row vector with containing co-occurence values

        for j in ss: #loop through all collumn vectors
            weight = round(wn.path_similarity(synset, j), 3)  #value obtained from find the shortest route though hyponym-hypernym relation from comparing two words
            data.append(weight)
        M.append(data)
    return M

values = co_occurrence(M, hyponyms) #calling function


Mnp = np.asarray(values)
Mnp = np.dot(Mnp, Mnp.T) #matrix multiplication with iteself and the transpose
coocc = Mnp


cooccurDict = {}
for i, row in enumerate(coocc):
    vectDict = {}
    for j, col in enumerate(coocc):
        vectDict[j] = round(coocc[i,j],3)
    cooccurDict[i] = vectDict

print('dictionary created')

model = glove.Glove(cooccurDict, d=100, alpha=0.75, x_max=100.0)
for epoch in range(25):
    err = model.train(batch_size=200, workers=9)
print("epoch %d, error %.3f" % (epoch, err), flush=True)

embeddings = model.W
embeddingsDict = {}
for i, vec in enumerate(model.W):
    embeddingsDict[labelHeaders[i]] = vec


tsne = TSNE(n_components=2, random_state=0)
reduced_data = tsne.fit_transform(embeddings)
# reduced_data = PCA(n_components=2).fit_transform(data)

kmeans = KMeans(init="k-means++", n_clusters=6, n_init=4)
kmeans.fit(reduced_data)

y_kmeans = kmeans.predict(reduced_data)
y_ksort = np.sort(y_kmeans)

colourmap = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])


plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colourmap[y_kmeans])
words = list(embeddingsDict.keys())

for label, x, y in zip(words, reduced_data[:, 0], reduced_data[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points",alpha=0.5)



centreLabel = {}
oldGroup = 1
tempList =[]

for i, group in enumerate(y_ksort):

    'print(i,group,hyponyms[i])'
    if oldGroup == group:
        tempList.append(hyponyms[i])
    if oldGroup != group:
        centreLabel[str(oldGroup)] = tempList
        tempList = []
        tempList.append(hyponyms[i])
    oldGroup = group
centreLabel[str(oldGroup)] = tempList


groupHyper = {i: [] for i in range(6)}

for i in range(len(centreLabel)):

    tempGroup = centreLabel.get(str(i))
    try:
        for index, this in enumerate(tempGroup):
            for that in tempGroup[index + 1:]:
                hypernym = this.lowest_common_hypernyms(that)
                groupHyper[i].append(hypernym)


    except TypeError as error:
        print(error)




centreTemp =[]
for i in groupHyper:
    print(centreLabel[str(i)])
    print(groupHyper[i])
    hypernymsToCount = (word for word in groupHyper[i])
    centreCount = Counter(tuple(item) for item in hypernymsToCount)
    centreTemp.append(centreCount.most_common(3))

centers = kmeans.cluster_centers_
colours = [i for i, element in enumerate(centers)]





Names = []
Scores = []
for i, element in enumerate(centreTemp):
    tempNames = []
    tempScores = []
    for c in element:
        tempNames.append(c[0][0].lemma_names()[0])
        tempScores.append(c[1])
    Names.append(tempNames)
    Scores.append(tempScores)

print(Names)
print(Scores)


clusters = []
coloursTest = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
coloursTestFixed = [ '#ff7f0e', '#1f77b4', '#8c564b', '#d62728', '#9467bd','#2ca02c']
plt.scatter(centers[:, 0], centers[:, 1], marker="X",edgecolors='black', s=169, linewidths=3, c=coloursTest, zorder=10)
# print([str(np.asarray(Names[0]).flatten())],[str(np.asarray(Scores[0]).flatten())])
for i, name in enumerate(Names):
    labels = ''
    for c, score in enumerate(Scores[i]):
        label = (name[c] + ' : ' + str(score))
        labels += (label + '\n')

    print(i,labels)
    clusterTemp = mpatches.Patch(color=coloursTestFixed[i], label=labels)
    clusters.append(clusterTemp)

plt.legend(handles=clusters)
plt.show()



