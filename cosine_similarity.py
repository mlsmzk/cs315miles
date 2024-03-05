from collections import Counter
import pandas as pd
import numpy as np
import string
import math
import matplotlib.pyplot as plt
import seaborn as sns

sentences = [
    "As spring unfolds, the warmth of the season encourages the first blossoms to open, signaling longer days ahead.",
    "Spring brings not only blooming flowers but also the anticipation of sunny days and outdoor activities.",
    "With the arrival of spring, people begin planning their summer vacations, eager to enjoy the seasonal warmth.",
    "The mild spring weather marks the transition from the cold winter to the inviting warmth of summer.",
    "During spring, families often start spending more time outdoors, enjoying the season's pleasant temperatures and the promise of summer fun.",
    "Summer continues the season's trend of growth and warmth, with gardens full of life and days filled with sunlight.",
    "The summer season is synonymous with outdoor adventures and enjoying the extended daylight hours that began in spring.",
    "As summer arrives, the warm weather invites a continuation of the outdoor activities that people began enjoying in spring.",
    "The transition into summer brings even warmer temperatures, allowing for beach visits and swimming, much awaited since the spring.",
    "Summer vacations are often planned as the days grow longer, a pattern that starts in the spring, culminating in peak summer leisure."
]

def cosine_similarity(v1, v2):
    if list(v1) != list(v2):
        dp = np.dot(v1, v2)
        return math.acos(dp / np.linalg.norm(v1) / np.linalg.norm(v2))
    return 0

if __name__ == "__main__":
    counts_per_sentence = {}
    for i, sentence in enumerate(sentences):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower().split(" ")
        counts_per_sentence['d' + str(i+1)] = Counter(sentence)

    df = pd.DataFrame.from_dict(counts_per_sentence).transpose()
    df.fillna(0, inplace=True)
    df = df.astype(int)
    # Now each sentence (row) is a vector and each column is a vector of word frequency
    # Each row, col pair represents the frequency at which the word appears in that sentence
    hm = pd.DataFrame(index=df.index, columns=df.index, dtype=float)
    hm.fillna(0, inplace=True)

    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i <= j:
                sim = cosine_similarity(row1,row2)
                hm.loc[i, j] = sim
                hm.loc[j, i] = sim

    print(hm)
    # Create heatmap using Seaborn
    maxval = hm.max().max()
    sns.heatmap(hm, vmin=0, vmax=maxval)

    # Show the plot
    plt.show()
