import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import sem, ttest_ind, ttest_rel

labels = pd.read_csv("CategoryLabels.txt", sep = " ")
vectors = pd.read_csv("CategoryVectors.txt")
responses = pd.read_csv("NeuralResponses_S1.txt")
responses_df = pd.DataFrame(data=responses)




animate = []
inanimate = []

for cat in range(len(vectors)):
    if vectors["Var1"][cat] == 1:
        animate.append(responses.iloc[cat])
    else:
        inanimate.append(responses.iloc[cat])


animate_df = pd.DataFrame(animate)
inanimate_df = pd.DataFrame(inanimate)


animate_avg = []
inanimate_avg = []


for idx, row in animate_df.iterrows():
    avg = np.mean(row)
    animate_avg.append(float(avg))

for idx, row in inanimate_df.iterrows():
    avg = np.mean(row)
    inanimate_avg.append(float(avg))

avg_anim = np.mean(animate_avg)
avg_inanim = np.mean(inanimate_avg)

cats = ["animate", "inanimate"]
avgs = [avg_anim, avg_inanim]

sems = [sem(animate_avg), sem(inanimate_avg)]

"""
plt.bar(cats, avgs)
plt.errorbar(cats, avgs, yerr=sems, ecolor = "red", fmt='none', capsize=5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.title("A. Average Response Amplitude")
plt.ylabel("Response Amplitude")
plt.show()
"""

def pairedTTest(m, s, n):
    return(m/(s/np.sqrt(n)))



difference = np.array(animate_avg) - np.array(inanimate_avg)
avg_diff = np.mean(difference)
std_diff = np.std(difference, ddof=1)

t = pairedTTest(avg_diff, std_diff, 44)

##print(f"t(43) = {t}, p = 0.12, this is not significant.")


animate_voxels = []
inanimate_voxels = []

for response in animate_df:
    animate_voxels.append(float(np.mean(animate_df[response])))

for response in inanimate_df:
    inanimate_voxels.append(float(np.mean(inanimate_df[response])))


voxelRange = np.arange(1,21)

diffs = np.array(animate_voxels) - np.array(inanimate_voxels)

diffsTwenty = diffs[:20]

"""
plt.bar(voxelRange, diffsTwenty)
plt.xticks(voxelRange, voxelRange.astype(int))
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.title("B. Animate minus Inanimate")
plt.ylabel("Response Amplitude")
plt.show()
"""

rdm_matrix = np.zeros((len(responses), len(responses)))


for a in range(len(responses)):
    for b in range(len(responses)):

        response_a = responses_df.iloc[a].values
        response_b = responses_df.iloc[b].values
        

        correlation = np.corrcoef(response_a, response_b)[0, 1]
        rdm_matrix[a][b] =  1 - correlation


plt.figure(figsize=(10, 8))
plt.imshow(rdm_matrix, cmap='OrRd', vmin=0.8, vmax=1.2)
plt.colorbar()
plt.title('A: RDM')
plt.tight_layout()
plt.show()


"""

animacy = vectors.iloc[:, 0].values  

animacy_mask = np.zeros((len(animacy), len(animacy)))

for i in range(len(animacy)):
    for j in range(len(animacy)):
        if animacy[i] == animacy[j]:
            animacy_mask[i, j] = 0  
        else:
            animacy_mask[i, j] = 1  


same_animacy_dissim = []
diff_animacy_dissim = []

for i in range(len(rdm_matrix)):
    for j in range(i + 1, len(rdm_matrix)):  
        if animacy_mask[i, j] == 0:  
            same_animacy_dissim.append(rdm_matrix[i, j])
        else: 
            diff_animacy_dissim.append(rdm_matrix[i, j])

same_animacy_dissim = np.array(same_animacy_dissim)
diff_animacy_dissim = np.array(diff_animacy_dissim)


t_stat, p_value = ttest_ind(same_animacy_dissim, diff_animacy_dissim)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6e}")

mean_same_dissim = np.mean(same_animacy_dissim)
mean_diff_dissim = np.mean(diff_animacy_dissim)
sem_same_dissim = np.std(same_animacy_dissim, ddof=1) / np.sqrt(len(same_animacy_dissim))
sem_diff_dissim = np.std(diff_animacy_dissim, ddof=1) / np.sqrt(len(diff_animacy_dissim))

print(f"Same animacy: mean = {mean_same_dissim:.4f}, SEM = {sem_same_dissim:.4f}")
print(f"Different animacy: mean = {mean_diff_dissim:.4f}, SEM = {sem_diff_dissim:.4f}")


fig, ax = plt.subplots(figsize=(6, 5))

categories = ['same', 'different']
means = [mean_same_dissim, mean_diff_dissim]
sems = [sem_same_dissim, sem_diff_dissim]

bars = ax.bar(categories, means, yerr=sems, capsize=10, 
               color='#4682b4', edgecolor='black', ecolor="orange", linewidth=1.5)

ax.set_ylabel('Dissimilarity', fontsize=12)
ax.set_title('B: Barplot', fontsize=13)
ax.set_ylim([0, 1.2])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
"""

## Neural Responses for S1
responses = pd.read_csv("NeuralResponses_S1.txt")
responses_df = pd.DataFrame(data=responses)
rdm_s1 = np.zeros((len(responses), len(responses)))

## RDM for S1
for a in range(len(responses)):
    for b in range(len(responses)):

        response_a = responses_df.iloc[a].values
        response_b = responses_df.iloc[b].values
        

        correlation = np.corrcoef(response_a, response_b)[0, 1]
        rdm_s1[a][b] =  1 - correlation

## Neural Responses for S2
responses_two = pd.read_csv("NeuralResponses_S2.txt")
responses_two_df = pd.DataFrame(data=responses_two)
rdm_s2 = np.zeros((len(responses_two), len(responses_two)))

## RDM for S2
for a in range(len(responses_two)):
    for b in range(len(responses_two)):

        response_a = responses_two_df.iloc[a].values
        response_b = responses_two_df.iloc[b].values
        

        correlation = np.corrcoef(response_a, response_b)[0, 1]
        rdm_s2[a][b] =  1 - correlation

upper_tri_indices = np.triu_indices(len(rdm_s1), k=1)

## Unique pairs extraction
s1_dissim = rdm_s1[upper_tri_indices]
s2_dissim = rdm_s2[upper_tri_indices]

## Correlation between S1 and S2 RDMs
corr_rdms = np.corrcoef(s1_dissim, s2_dissim)[0, 1]
print(f"\n(1) Correlation (all categories): {corr_rdms:.4f}")

## Dissimilarity between S1 and S2 RDMs
avg_dissimilarity = np.mean(np.abs(s1_dissim - s2_dissim))
print(f"Average dissimilarity: {avg_dissimilarity:.4f}")

## Paired t-test: are S1 and S2 dissimilarities significantly different?
t_stat, p_value = ttest_rel(s1_dissim, s2_dissim)
print(f"t-value: {t_stat:.4f}")
print(f"p-value: {p_value:.6e}")













"""
animate_pairs = []
for i in range(44):  
    for j in range(i + 1, 44):  
        animate_pairs.append((i, j))

neural_animate = [rdm_matrix[i, j] for i, j in animate_pairs]
behavior_animate = [behavior_rdm[i, j] for i, j in animate_pairs]

corr_animate = np.corrcoef(neural_animate, behavior_animate)[0, 1]
print(f"(2) Correlation (animate only): {corr_animate:.4f}")


inanimate_pairs = []
for i in range(44, 88):  
    for j in range(i + 1, 88):  
        inanimate_pairs.append((i, j))

neural_inanimate = [rdm_matrix[i, j] for i, j in inanimate_pairs]
behavior_inanimate = [behavior_rdm[i, j] for i, j in inanimate_pairs]

corr_inanimate = np.corrcoef(neural_inanimate, behavior_inanimate)[0, 1]
print(f"(3) Correlation (inanimate only): {corr_inanimate:.4f}")

same_animacy_dissim = []
diff_animacy_dissim = []

for i in range(len(behavior_rdm)):
    for j in range(i + 1, len(behavior_rdm)):  
        if animacy_mask[i, j] == 0:  
            same_animacy_dissim.append(behavior_rdm[i, j])
        else: 
            diff_animacy_dissim.append(behavior_rdm[i, j])

same_animacy_dissim = np.array(same_animacy_dissim)
diff_animacy_dissim = np.array(diff_animacy_dissim)


t_stat, p_value = ttest_ind(same_animacy_dissim, diff_animacy_dissim)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6e}")

mean_same_dissim = np.mean(same_animacy_dissim)
mean_diff_dissim = np.mean(diff_animacy_dissim)
sem_same_dissim = np.std(same_animacy_dissim, ddof=1) / np.sqrt(len(same_animacy_dissim))
sem_diff_dissim = np.std(diff_animacy_dissim, ddof=1) / np.sqrt(len(diff_animacy_dissim))

print(f"Same animacy: mean = {mean_same_dissim:.4f}, SEM = {sem_same_dissim:.4f}")
print(f"Different animacy: mean = {mean_diff_dissim:.4f}, SEM = {sem_diff_dissim:.4f}")
"""