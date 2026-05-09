from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = self.kernel_size
        self.W = initialize_method(size=(out_channels, in_channels, kh, kw))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        self.input = None
        self.input_padded = None
        self.cols = None
        self.output_hw = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def _im2col(self, X, out_h, out_w):
        batch, channels, _, _ = X.shape
        kh, kw = self.kernel_size
        cols = np.empty((batch, channels, kh, kw, out_h, out_w), dtype=X.dtype)
        for i in range(kh):
            i_max = i + self.stride * out_h
            for j in range(kw):
                j_max = j + self.stride * out_w
                cols[:, :, i, j, :, :] = X[:, :, i:i_max:self.stride, j:j_max:self.stride]
        cols = cols.transpose(0, 4, 5, 1, 2, 3)
        return cols.reshape(batch * out_h * out_w, -1)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        assert X.ndim == 4
        assert X.shape[1] == self.in_channels

        self.input = X
        if self.padding > 0:
            X_padded = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        else:
            X_padded = X
        self.input_padded = X_padded

        _, _, height, width = X_padded.shape
        kh, kw = self.kernel_size
        out_h = (height - kh) // self.stride + 1
        out_w = (width - kw) // self.stride + 1
        assert out_h > 0 and out_w > 0
        self.output_hw = (out_h, out_w)

        self.cols = self._im2col(X_padded, out_h, out_w)
        W_col = self.W.reshape(self.out_channels, -1)
        output = self.cols @ W_col.T
        output = output.reshape(X.shape[0], out_h, out_w, self.out_channels)
        output = output.transpose(0, 3, 1, 2)
        return output + self.b

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch = grads.shape[0]
        out_h, out_w = self.output_hw
        kh, kw = self.kernel_size
        assert grads.shape == (batch, self.out_channels, out_h, out_w)

        grads_col = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        W_col = self.W.reshape(self.out_channels, -1)

        self.grads['W'] = (grads_col.T @ self.cols).reshape(self.W.shape)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        dcols = grads_col @ W_col
        dcols = dcols.reshape(batch, out_h, out_w, self.in_channels, kh, kw)
        dcols = dcols.transpose(0, 3, 4, 5, 1, 2)

        dX_padded = np.zeros_like(self.input_padded)
        for i in range(kh):
            i_max = i + self.stride * out_h
            for j in range(kw):
                j_max = j + self.stride * out_w
                dX_padded[:, :, i:i_max:self.stride, j:j_max:self.stride] += dcols[:, :, i, j, :, :]

        if self.padding > 0:
            return dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class Dropout(Layer):
    """
    Inverted dropout. During training, activations are masked and scaled by
    1 / (1 - p). During evaluation, dropout is an identity mapping.
    """
    def __init__(self, p=0.5) -> None:
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.training = True
        self.mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def set_training(self, training=True):
        self.training = training

    def forward(self, X):
        if (not self.training) or self.p == 0:
            self.mask = None
            return X
        keep_prob = 1.0 - self.p
        self.mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
        return X * self.mask

    def backward(self, grads):
        if self.mask is None:
            return grads
        return grads * self.mask

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None

        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels.astype(np.int64)

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        batch_size = predicts.shape[0]
        idx = np.arange(batch_size)
        eps = 1e-12
        loss = -np.log(np.clip(self.probs[idx, self.labels], eps, 1.0)).mean()
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        idx = np.arange(batch_size)
        eps = 1e-12

        if self.has_softmax:
            grads = self.probs.copy()
            grads[idx, self.labels] -= 1.0
            grads /= batch_size
        else:
            grads = np.zeros_like(self.predicts)
            grads[idx, self.labels] = -1.0 / np.clip(self.predicts[idx, self.labels], eps, 1.0)
            grads /= batch_size

        self.grads = grads
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
