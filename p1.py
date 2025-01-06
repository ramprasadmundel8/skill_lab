import numpy as np
def compute_tf_idf(documents, vocabulary):
    N = len(documents)
    V = len(vocabulary)
    tf = np.zeros((N, V))
    for i, doc in enumerate(documents):
        words = doc.lower().split()
        for word in words:
            if word in vocabulary:
                j = vocabulary.index(word)
                tf[i, j] += 1
        tf[i] = tf[i] / len(words)

    df = np.zeros(V)
    for j, term in enumerate(vocabulary):
        df[j] = sum(1 for doc in documents if term in doc.lower().split())

    idf = np.log(N / (df + 1))
    tf_idf = tf * idf
    return tf_idf
documents = [
    "cat sat on the mat",
    "dog sat on the log",
    "cat and dog played together"
]
vocabulary = list(set(" ".join(documents).lower().split()))
tf_idf_matrix = compute_tf_idf(documents, vocabulary)
print("Vocabulary:", vocabulary)
print("TF-IDF Matrix:\n", tf_idf_matrix)
