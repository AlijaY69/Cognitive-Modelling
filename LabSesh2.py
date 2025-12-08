import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

category_vectors = pd.read_csv("FilesForAssignment2/CategoryVectors.txt")
category_labels = pd.read_csv("FilesForAssignment2/CategoryLabels.txt", sep= " ")
# print(category_labels.head())
# print(category_vectors.head())

neuralresponse_1 = pd.read_csv("FilesForAssignment2/NeuralResponses_S1.txt")
neuralresponse_2 = pd.read_csv("FilesForAssignment2/NeuralResponses_S2.txt")

responses_animate = neuralresponse_1[category_vectors["Var1"] == 1].copy()
responses_inanimate = neuralresponse_1[category_vectors["Var2"] == 1].copy()

mean_responses_animate = responses_animate.mean(1)
mean_responses_inanimate = responses_inanimate.mean(1)

mean_responses_animate = mean_responses_animate.reset_index(drop=True)
mean_responses_inanimate = mean_responses_inanimate.reset_index(drop=True)

# BRIGHTSPACE Q 1A:
#print(f"Animate: {mean_responses_animate}, Inanimate: {mean_responses_inanimate}")
# plt.bar(["Animate", "Inanimate"], [mean_responses_animate.mean(), mean_responses_inanimate.mean()], 
#         yerr = [mean_responses_animate.sem(), mean_responses_inanimate.sem()], ecolor= "red", capsize = 3, edgecolor = "black")
# plt.ylabel("Response Amplitude")
# plt.title("Average Response Amplitude")
# plt.show()

difference_responses = mean_responses_animate - mean_responses_inanimate

def paired_t_test(m, s, n):
    return m/(s/sqrt(n))

# BRIGHTSPACE Q 1B:
# print(paired_t_test(difference_responses.mean(), difference_responses.std(), difference_responses.shape[0]))
# prints: 1.6013306249680146
# With ttable.org we see a confidence level of roughly 0.95, DF = 43


mean_voxels_animate = responses_animate.mean(0)
mean_voxels_inanimate = responses_inanimate.mean(0)

mean_voxels_animate = mean_voxels_animate.reset_index(drop=True)
mean_voxels_inanimate = mean_voxels_inanimate.reset_index(drop=True)

difference_voxels = mean_voxels_animate - mean_voxels_inanimate

# BRIGHTSPACE Q 1C & 1D:
# plt.bar(range(20), difference_voxels.iloc[:20], ecolor= "red", capsize = 3, edgecolor = "black")
# plt.ylabel("Animate Minus Inanimate")
# plt.title("Response Amplitude")
# plt.xticks(range(20))
# plt.show()

responses_animate_2 = neuralresponse_2[category_vectors["Var1"] == 1].copy()
responses_inanimate_2 = neuralresponse_2[category_vectors["Var2"] == 1].copy()

# sanity check
# print(responses_animate_2.shape)
# print(responses_inanimate_2.shape)

X_train = pd.concat([responses_animate_2.iloc[:22], responses_inanimate_2.iloc[:22]])
X_test = pd.concat([responses_animate_2.iloc[22:], responses_inanimate_2.iloc[22:]])

y_train = [1]*22 + [-1]*22
y_test =  [1]*22 + [-1]*22

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# BRIGHTSPACE Q 2A:
# print(accuracy_score(y_test, y_pred))
# is around 0.7 most of the time

weights = clf.coef_[0, :20]
voxels = X_train.mean(0)[:20]

# BRIGHTSPACE Q 2B:
# plt.scatter(weights, voxels)
# plt.ylabel("Responses")
# plt.xlabel("Weights")
# plt.show()

pearson = np.corrcoef(weights, voxels)[0, 1]

# BRIGHTSPACE Q 2C:
# print(pearson)

responses_inhuman = responses_animate_2[category_vectors["Var4"] == 1].copy()
responses_human = responses_animate_2[category_vectors["Var3"] == 1].copy()
responses_human = responses_human.iloc[:20]

X_train_2 = pd.concat([responses_human.iloc[:10], responses_inhuman.iloc[:10]])
X_test_2 = pd.concat([responses_human.iloc[10:], responses_inhuman.iloc[10:]])

y_train_2 = [1]*10 + [-1]*10
y_test_2 =  [1]*10 + [-1]*10

clf_2 = svm.SVC(kernel='linear')
clf_2.fit(X_train_2, y_train_2)
y_pred_2 = clf_2.predict(X_test_2)

# BRIGHTSPACE Q 2D:
# ?