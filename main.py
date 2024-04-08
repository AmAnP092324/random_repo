import pickle, os, numpy as np

sigmoid = lambda x: 1/(1 + np.exp(-x))
tanh = lambda x: np.tanh(x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

def squared_error(yhat, y):
    loss = np.sum(np.dot(y-yhat, (y-yhat).T), axis=0, keepdims=True)
    return loss

def cross_entropy_loss(yhat, y):
    yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
    loss = -np.sum(y * np.log(yhat), axis=1, keepdims=True)
    return loss

class Layer:
    def __init__(self, input_size, output_size, activation, opt, momentum, lr):
        self.input_size = input_size
        self.output_size = output_size
        self.opt = opt
        self.momentum = momentum
        self.lr = lr
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((output_size, 1))
        if activation == 'sigmoid':
            self.activation = sigmoid
        elif activation == 'tanh':
            self.activation = tanh
        self.vW = np.zeros_like(self.weights)
        self.vb = np.zeros_like(self.biases)

    def forward(self, X):
        self.input = X
        self.z = np.dot(self.weights.T, X) + self.biases
        self.output = self.activation(self.z)
        return self.output

    def backward(self, delta, lr):
        if self.activation == sigmoid:
            delta *= sigmoid_derivative(self.z)
        elif self.activation == tanh:
            delta *= 1 - np.square(tanh(self.z))

        self.dW = np.dot(delta, self.input.T).T / self.input.shape[0]
        self.db = np.mean(delta, axis=1)
        self.db = np.mean(self.db, axis=0)
        if self.opt == "gd":
            delta = np.dot(self.weights, delta)
            self.weights -= lr * self.dW
            self.biases -= lr * self.db
            return delta
        elif self.opt == "momentum":
            self.vW = self.momentum * self.vW - self.lr * self.dW
            self.vb = self.momentum * self.vb - self.lr * self.db
            self.weights += self.vW
            self.biases += self.vb

            delta = np.dot(delta, self.weights.T)
            return delta
        elif self.opt == "nag":
            vW_prev = self.vW
            vb_prev = self.vb
            self.vW = self.momentum * self.vW - self.lr * self.dW
            self.vb = self.momentum * self.vb - self.lr * self.db

            self.weights += -self.momentum * vW_prev + (1 + self.momentum) * self.vW
            self.biases += -self.momentum * vb_prev + (1 + self.momentum) * self.vb

            delta = np.dot(delta, self.weights.T)
            return delta
        elif self.opt == "adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.mW = np.zeros_like(self.weights)
            self.mb = np.zeros_like(self.biases)
            self.vW = np.zeros_like(self.weights)
            self.vb = np.zeros_like(self.biases)
            self.t = 0

            self.t += 1
            self.mW = self.beta1 * self.mW + (1 - self.beta1) * self.dW
            self.mb = self.beta1 * self.mb + (1 - self.beta1) * self.db
            self.vW = self.beta2 * self.vW + (1 - self.beta2) * np.square(self.dW)
            self.vb = self.beta2 * self.vb + (1 - self.beta2) * np.square(self.db)

            # Bias-corrected moments
            mW_corrected = self.mW / (1 - self.beta1 ** self.t)
            mb_corrected = self.mb / (1 - self.beta1 ** self.t)
            vW_corrected = self.vW / (1 - self.beta2 ** self.t)
            vb_corrected = self.vb / (1 - self.beta2 ** self.t)

            # Update weights and biases using Adam
            self.weights -= self.lr * mW_corrected / (np.sqrt(vW_corrected) + self.epsilon)
            self.biases -= self.lr * mb_corrected / (np.sqrt(vb_corrected) + self.epsilon)

            delta = np.dot(delta, self.weights.T)
            return delta
        

class NeuralNetwork:
    def __init__(self, momentum, num_hidden, sizes, activation, lr, loss, opt):
        self.momentum = momentum
        self.hidden_layers = num_hidden
        self.sizes = sizes
        self.activation = activation
        self.layers = []
        self.lr = lr
        if loss=="ce":
            self.loss = cross_entropy_loss
        elif loss == "sq":
            self.loss = squared_error

        layer_input_size = 3072 # 32x32x3
        for i in range(self.hidden_layers):
            layer_output_size = sizes[i]
            layer = Layer(layer_input_size, layer_output_size, activation, opt, momentum, lr)
            self.layers.append(layer)
            layer_input_size = layer_output_size
        output_layer = Layer(layer_input_size, 10, activation, opt, momentum, lr)
        self.layers.append(output_layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta, lr):
        for layer in reversed(self.layers):
            delta = layer.backward(delta, lr)

    def train(self, X_train, y_train, epochs, batch_size, anneal=False, X_val=None, y_val=None):
        num_samples = X_train.shape[0]
        num_steps = num_samples // batch_size


        for epoch in range(epochs):
            if anneal and epoch > 0 and epoch % 2 == 0:
                self.lr /= 2

            permutation = np.random.permutation(num_samples)
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            for step in range(num_steps):
                start = step * batch_size
                end = start + batch_size
                X_batch = ((X_train.T)[start:end]).T
                y_batch = (y_train[start:end]).T

                y_pred = self.forward(X_batch)
                loss = self.loss(y_pred, y_batch)
                error_rate = self.compute_error_rate(y_pred, y_batch)

                delta = y_pred - y_batch
                self.backward(delta, self.lr)

                if (step + 1) % 100 == 0:
                    print(f"Epoch {epoch}, Step {step + 1}, Loss: {loss}, Error: {error_rate}, lr: {self.lr}")

            # Compute loss and error rate on validation data
            if X_val and y_val:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss(y_val_pred, y_val)
                val_error_rate = self.compute_error_rate(y_val_pred, y_val)
                file = open("log train.txt", 'w+')
                file.write(f"Epoch {epoch}, Validation Loss: {val_loss}, Validation Error: {val_error_rate}\n")
                file.close()

    def compute_error_rate(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return np.mean(y_pred != y_true)

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "model.pkl"), "wb") as file:
            pickle.dump(self, file)

    def validate(self, X_val, y_val):
        y_val_pred = self.forward(X_val)
        val_loss = self.loss(y_val_pred, y_val.T)
        val_error_rate = self.compute_error_rate(y_val_pred, y_val)
        print(f"Validation Loss: {val_loss}, Validation Error: {val_error_rate}")


    @staticmethod
    def load_model(save_dir):
        with open(os.path.join(save_dir, "model.pkl"), "rb") as file:
            return pickle.load(file)



# def normalize_rows(x, axis):
#     # x=1 rows, x=0 columns
#     x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
#     x = x / x_norm

#     return x
    
# def softmax(x):
#     x_exp = np.exp(x)
#     x_sum = np.sum(x_exp, axis=1, keepdims= True)
#     s = x_exp / x_sum
    
#     return s

# image2vector = lambda img: img.reshape((img.shape[0]*img.shape[1]*img.shape[2], 1))
