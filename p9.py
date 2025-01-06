import numpy as np
def rnn_forward(x, Wxh, Whh, Why, bh, by, h0):
    h = h0
    hs = []
    ys = []
    for t in range(len(x)):
        xt = np.array([[x[t]]])
        h = np.tanh(np.dot(Whh, h) + np.dot(Wxh, xt) + bh)
        y = np.dot(Why, h) + by
        hs.append(h)
        ys.append(y)
    return ys, hs
x = [1, 2, 3]
input_size = 1
hidden_size = 4
output_size = 1

np.random.seed(0)
Wxh = np.random.randn(hidden_size, input_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(output_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))
h0 = np.zeros((hidden_size, 1))
ys, hs = rnn_forward(x, Wxh, Whh, Why, bh, by, h0)

print("Outputs at each time step:")
for t, y in enumerate(ys):
    print(f"Time step {t+1}: y = {y.flatten()}")
