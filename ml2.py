import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import sem
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

labels = pd.read_csv("CategoryLabels.txt", sep = " ")
vectors = pd.read_csv("CategoryVectors.txt")
responses = pd.read_csv("NeuralResponses_S2.txt")
responses_df = pd.DataFrame(data=responses)

animate = []
inanimate = []

for cat in range(len(vectors)):
    if vectors["Var1"][cat] == 1:
        animate.append(responses.iloc[cat])
    else:
        inanimate.append(responses.iloc[cat])

animate_df = pd.DataFrame(animate)

animate_train = []
animate_test = []
inanimate_train = []
inanimate_test = []

for tr in range(len(animate)):
    if tr < 22:
        animate_train.append(animate[tr])
    else:
        animate_test.append(animate[tr])


for tr in range(len(inanimate)):
    if tr < 22:
        inanimate_train.append(inanimate[tr])
    else:
        inanimate_test.append(inanimate[tr])

animate_train_df = pd.DataFrame(animate_train)
animate_test_df = pd.DataFrame(animate_test)
inanimate_train_df = pd.DataFrame(inanimate_train)
inanimate_test_df = pd.DataFrame(inanimate_test)

animate_train_df['label'] = 1
animate_test_df['label'] = 1
inanimate_train_df['label'] = -1
inanimate_test_df['label'] = -1

train_set = pd.concat([animate_train_df, inanimate_train_df])
test_set = pd.concat([animate_test_df, inanimate_test_df])

X = train_set.iloc[:, :-1] 
y = train_set.iloc[:, -1]

clf = SVC(kernel = 'linear', C=10)
clf.fit(X, y)
prediction = clf.predict(test_set.iloc[:, :-1])

weights = clf.coef_

weights_twenty = weights[0, :20]

responses_twenty = responses[:20]


animate_mean = animate_train_df.iloc[:, :20].mean(axis=0).values
inanimate_mean = inanimate_train_df.iloc[:, :20].mean(axis=0).values
response_diff = animate_mean - inanimate_mean

plt.scatter(weights_twenty, response_diff)
plt.xlabel("Weights")
plt.ylabel("Responses")
plt.title("Scatter Plot")
## plt.show()

## print(np.corrcoef(weights_twenty, response_diff)[0][1])

human_train = animate_df[:10].copy()
human_test = animate_df[10:20].copy()

inhuman_train = animate_df[24:34].copy()
inhuman_test = animate_df[34:44].copy()

human_train['label'] = 1
human_test['label'] = 1
inhuman_train['label'] = -1
inhuman_test['label'] = -1


train_s = pd.concat([human_train, inhuman_train])
test_s = pd.concat([human_test, inhuman_test])

X2 = train_s.iloc[:, :-1] 
y2 = train_s.iloc[:, -1]

clf = SVC(kernel = 'linear', C=10)
clf.fit(X2, y2)
prediction = clf.predict(test_s.iloc[:, :-1])

print(accuracy_score(prediction, test_s.iloc[:,-1]))