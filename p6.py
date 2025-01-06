import numpy as np
def create_one_hot_encodings(corpus):
    vocabulary = {}
    index = 0
    for sentence in corpus:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    V = len(vocabulary)
    one_hot_encodings = {}
    for word, idx in vocabulary.items():
        one_hot_vector = np.zeros(V)
        one_hot_vector[idx] = 1
        one_hot_encodings[word] = one_hot_vector

    return vocabulary, one_hot_encodings
corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]

vocabulary, one_hot_encodings = create_one_hot_encodings(corpus)

print("Vocabulary:", vocabulary)
print("\nOne-Hot Encodings:")
for word, one_hot_vector in one_hot_encodings.items():
    print(f"Word: '{word}'- One-Hot Vector: {one_hot_vector}")
