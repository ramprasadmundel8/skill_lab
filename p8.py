def generate_cbow_pairs(sentences, window_size):
    vocabulary = {}
    index = 0
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    training_pairs = []
    for sentence in sentences:
        words = sentence.lower().split()
        for i, target_word in enumerate(words):
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            context_words = []
            for j in range(start, end):
                if i != j:
                    context_words.append(words[j])
            if context_words:
                training_pairs.append((tuple(context_words), target_word))

    return vocabulary, training_pairs
sentences = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]
window_size = 2
vocabulary, training_pairs = generate_cbow_pairs(sentences, window_size)

print("Vocabulary:", vocabulary)
print("\nCBOW Training Pairs:")
for pair in training_pairs:
    print(f"Context: {pair[0]}, Target: {pair[1]}")
