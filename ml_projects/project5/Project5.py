import numpy as np
import matplotlib.pyplot as plt

np.random.seed(50)

#################################
# Project-5 a) XOR classification
#################################

class OneHiddenLayerXOR:
    def __init__(self, n_hidden=2, lr=0.1, n_iters=10000):
        self.n_hidden  = n_hidden
        self.lr        = lr
        self.n_iters   = n_iters
        self.loss_hist = []

    def _tanh(self, z):
        return np.tanh(z)

    def _compute_loss(self, y_hat, y):
        eps   = 1e-15
        y_hat = np.clip(y_hat, eps, 1-eps)
        return -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

    def fit(self, X, y):
        m, d = X.shape
        y = y.reshape(-1, 1)

        # initialize parameters
        self.W1 = np.random.randn(d, self.n_hidden) * 0.1
        self.b1 = np.zeros((1, self.n_hidden))
        self.W2 = np.random.randn(self.n_hidden, 1) * 0.1
        self.b2 = np.zeros((1, 1))

        self.loss_hist = []
        for epoch in range(self.n_iters):
            # forward
            Z1 = X.dot(self.W1) + self.b1
            A1 = self._tanh(Z1)
            Z2 = A1.dot(self.W2) + self.b2
            A2 = self._tanh(Z2)

            # loss
            loss = self._compute_loss(A2, y)
            self.loss_hist.append(loss)

            # backward
            dZ2 = A2 - y
            dW2 = (1/m) * A1.T.dot(dZ2)
            db2 = (1/m) * np.sum(dZ2, axis=0)
            dA1 = dZ2.dot(self.W2.T)
            dZ1 = dA1 * (1 - A1**2)
            dW1 = (1/m) * X.T.dot(dZ1)
            db1 = (1/m) * np.sum(dZ1, axis=0)

            # update
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2.reshape(1,1)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1.reshape(1,self.n_hidden)

    def predict(self, X):
        A1 = self._tanh(X.dot(self.W1) + self.b1)
        A2 = self._tanh(A1.dot(self.W2) + self.b2)
        return (A2 >= 0.5).astype(int).flatten()


# --- run part a) ---
X_mat  = np.array([[-1,  1, -1,  1],
                   [-1,  1,  1, -1]])
X_xor  = X_mat.T
y_xor  = np.array([0, 0, 1, 1])

xor_model = OneHiddenLayerXOR(n_hidden=3, lr=0.9, n_iters=1500)
xor_model.fit(X_xor, y_xor)

y_pred_xor = xor_model.predict(X_xor)
print("y_real     =", y_xor)
print("y_predicted=", y_pred_xor)

# static plot of training loss (log scale)
plt.figure()
plt.yscale('log')
plt.plot(xor_model.loss_hist)
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss (log scale)')
plt.title('XOR Training Loss')
plt.grid(True)

# static 3D decision boundary
xx, yy = np.meshgrid(
    np.linspace(X_xor[:,0].min()-1, X_xor[:,0].max()+1, 100),
    np.linspace(X_xor[:,1].min()-1, X_xor[:,1].max()+1, 100)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z    = xor_model.predict(grid).reshape(xx.shape)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap='coolwarm', alpha=0.6, edgecolor='k')
ax.scatter(X_xor[:,0], X_xor[:,1], y_xor,
           c=y_xor, cmap='coolwarm', edgecolors='k', s=80)
ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_zlabel('Class (0/1)')
ax.set_title('3D XOR Decision Boundary')


##################################
# Project-5 b) Regression
##################################

class FFNNRegressor:
    """
    Feed-forward NN for regression with:
      - arbitrary # of hidden layers
      - arbitrary units per hidden layer
      - tanh/relu/sigmoid hidden activations
      - mini-batch gradient descent
      - MSE loss
    """
    def __init__(self, input_dim, hidden_layers=1, hidden_units=10,
                 hidden_activation='tanh', lr=0.01, n_iters=2000,
                 batch_size=None):
        self.hidden_layers     = hidden_layers
        self.hidden_units      = hidden_units
        self.hidden_activation = hidden_activation
        self.lr                = lr
        self.n_iters           = n_iters
        self.batch_size        = batch_size

        # choose activation
        if hidden_activation == 'tanh':
            self.act      = np.tanh
            self.act_grad = lambda a: 1 - a**2
        elif hidden_activation == 'relu':
            self.act      = lambda z: np.maximum(0, z)
            self.act_grad = lambda a: (a > 0).astype(float)
        elif hidden_activation == 'sigmoid':
            self.act      = lambda z: 1/(1+np.exp(-z))
            self.act_grad = lambda a: a*(1 - a)
        else:
            raise ValueError("hidden_activation must be 'tanh','relu' or 'sigmoid'")

        # initialize weights & biases
        dims = [input_dim] + [hidden_units]*hidden_layers + [1]
        self.weights = [
            np.random.randn(dims[i], dims[i+1]) * 0.1
            for i in range(len(dims)-1)
        ]
        self.biases  = [
            np.zeros((1, dims[i+1]))
            for i in range(len(dims)-1)
        ]

        self.loss_history = []

    def _mse_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def fit(self, X, y):
        m = X.shape[0]
        y = y.reshape(m, 1)
        self.loss_history = []

        for epoch in range(self.n_iters):
            # shuffle
            perm    = np.random.permutation(m)
            Xs, ys  = X[perm], y[perm]
            bs      = self.batch_size or m

            # mini-batch updates
            for start in range(0, m, bs):
                end = start + bs
                Xb  = Xs[start:end]
                yb  = ys[start:end]
                mb  = Xb.shape[0]

                # forward
                A = Xb
                acts = [A]
                for l in range(self.hidden_layers):
                    Z = A.dot(self.weights[l]) + self.biases[l]
                    A = self.act(Z)
                    acts.append(A)
                Z_out = A.dot(self.weights[-1]) + self.biases[-1]
                A_out = Z_out
                acts.append(A_out)

                # backward
                grads_W = [None]*len(self.weights)
                grads_b = [None]*len(self.biases)

                dA = 2*(A_out - yb)/mb
                dZ = dA
                grads_W[-1] = acts[-2].T.dot(dZ)
                grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

                for l in range(self.hidden_layers-1, -1, -1):
                    dA = dZ.dot(self.weights[l+1].T)
                    dZ = dA * self.act_grad(acts[l+1])
                    grads_W[l] = acts[l].T.dot(dZ)
                    grads_b[l] = np.sum(dZ, axis=0, keepdims=True)

                # update
                for l in range(len(self.weights)):
                    self.weights[l] -= self.lr * grads_W[l]
                    self.biases[l]  -= self.lr * grads_b[l]

            # record full-dataset loss
            y_full = self.predict(X).reshape(m,1)
            self.loss_history.append(self._mse_loss(y_full, y))

    def predict(self, X):
        A = X
        for l in range(self.hidden_layers):
            A = self.act(A.dot(self.weights[l]) + self.biases[l])
        return (A.dot(self.weights[-1]) + self.biases[-1]).flatten()


# --- run part b) ---
np.random.seed(100)
X_reg = 2 * np.random.rand(50,1) - 1
y_reg = np.sin(2*np.pi*X_reg.flatten()) + 0.3*np.random.randn(50)

configs = [
    (3, 5000),
    (20, 5000),
]

for units, iters in configs:
    model = FFNNRegressor(
        input_dim=1,
        hidden_layers=1,
        hidden_units=units,
        hidden_activation='tanh',
        lr=0.1,
        n_iters=iters,
        batch_size=32
    )
    model.fit(X_reg, y_reg)

    # static MSE loss plot
    plt.figure()
    plt.plot(model.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{units} tanh units – Training Loss')
    plt.grid(True)

    # static data vs. fit plot
    y_pred = model.predict(X_reg)
    rmse   = np.sqrt(np.mean((y_pred - y_reg)**2))
    idx    = np.argsort(X_reg.flatten())
    Xs     = X_reg.flatten()[idx]
    preds  = y_pred[idx]

    plt.figure()
    plt.scatter(X_reg, y_reg, label='Data', s=30)
    plt.plot(Xs, preds, 'r-', label='Model')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'{units} tanh units – Fit (RMSE={rmse:.3f})')
    plt.legend()
    plt.grid(True)

# finally block until all plots are closed
plt.show()
