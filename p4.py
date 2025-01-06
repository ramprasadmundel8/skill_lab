import numpy as np
def create_embedding_matrix(corpus, embedding_dim):
    vocabulary = {}
    index = 0
    for sentence in corpus:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    V = len(vocabulary)
    E = np.random.rand(V, embedding_dim)
    word_to_index = vocabulary

    def get_word_vector(word):
        word = word.lower()
        if word in word_to_index:
            idx = word_to_index[word]
            return E[idx]
        else:
            return np.zeros(embedding_dim)

    return E, vocabulary, get_word_vector

corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]
embedding_dim = 3
E, vocabulary, get_word_vector = create_embedding_matrix(corpus, embedding_dim)
print("Vocabulary:", vocabulary)
print("Embedding Matrix E:\n", E)
word = "learning"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)
word = "unknown"
vector = get_word_vector(word)
print(f"Embedding for '{word}':", vector)
