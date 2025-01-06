def generate_skip_gram_pairs(sentences, window_size):
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
            # Define the context window
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_word = words[j]
                    training_pairs.append((target_word, context_word))

    return vocabulary, training_pairs
sentences = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love learning new things"
]
window_size = 2
vocabulary, training_pairs = generate_skip_gram_pairs(sentences, window_size)
print("Vocabulary:", vocabulary)
print("\nSkip-Gram Training Pairs:")
for pair in training_pairs:
    print(pair)
