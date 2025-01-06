import numpy as np
def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
def self_attention(X, Wq, Wk, Wv):
    Q = np.dot(X, Wq)
    K = np.dot(X, Wk)
    V = np.dot(X, Wv)
    d_k = Q.shape[1]
    attention_scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(attention_scores, axis=-1)
    output = np.dot(attention_weights, V)
    return output
np.random.seed(0)
X = np.random.rand(4, 3)
d = 3
dout = 2
Wq = np.random.rand(d, dout)
Wk = np.random.rand(d, dout)
Wv = np.random.rand(d, dout)
output = self_attention(X, Wq, Wk, Wv)
print("Input Matrix X:")
print(X)
print("\nWeight Matrix Wq:")
print(Wq)
print("\nWeight Matrix Wk:")
print(Wk)
print("\nWeight Matrix Wv:")
print(Wv)
print("\nSelf-Attention Output:")
print(output)
