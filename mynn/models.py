from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(
        self,
        in_channels=1,
        conv_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        num_classes=10,
        input_hw=(28, 28),
        hidden_dim=64,
        dropout_rate=0.0,
        weight_decay=False,
        weight_decay_lambda=1e-4,
    ):
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_classes = num_classes
        self.input_hw = input_hw
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        out_h = (input_hw[0] + 2 * padding - kernel_size) // stride + 1
        out_w = (input_hw[1] + 2 * padding - kernel_size) // stride + 1
        self.conv_output_shape = (conv_channels, out_h, out_w)
        self.flatten_dim = conv_channels * out_h * out_w

        self.conv = conv2D(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_decay=weight_decay,
            weight_decay_lambda=weight_decay_lambda,
        )
        self.relu = ReLU()
        self.fc1 = Linear(
            in_dim=self.flatten_dim,
            out_dim=hidden_dim,
            weight_decay=weight_decay,
            weight_decay_lambda=weight_decay_lambda,
        )
        self.relu_fc = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(
            in_dim=hidden_dim,
            out_dim=num_classes,
            weight_decay=weight_decay,
            weight_decay_lambda=weight_decay_lambda,
        )
        self.layers = [self.conv, self.relu, self.fc1, self.relu_fc, self.dropout, self.fc2]
        self._conv_activated_shape = None

    def __call__(self, X):
        return self.forward(X)

    def set_training(self, training=True):
        self.dropout.set_training(training)

    def forward(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.in_channels, self.input_hw[0], self.input_hw[1])
        elif X.ndim == 3:
            X = X.reshape(X.shape[0], self.in_channels, X.shape[1], X.shape[2])
        outputs = self.conv(X)
        outputs = self.relu(outputs)
        self._conv_activated_shape = outputs.shape
        outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = self.fc1(outputs)
        outputs = self.relu_fc(outputs)
        outputs = self.dropout(outputs)
        return self.fc2(outputs)

    def backward(self, loss_grad):
        grads = self.fc2.backward(loss_grad)
        grads = self.dropout.backward(grads)
        grads = self.relu_fc.backward(grads)
        grads = self.fc1.backward(grads)
        grads = grads.reshape(self._conv_activated_shape)
        grads = self.relu.backward(grads)
        return self.conv.backward(grads)
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        config = param_list['config']
        self.__init__(**config)

        self.conv.W = param_list['conv']['W']
        self.conv.b = param_list['conv']['b']
        self.conv.params['W'] = self.conv.W
        self.conv.params['b'] = self.conv.b
        self.conv.weight_decay = param_list['conv']['weight_decay']
        self.conv.weight_decay_lambda = param_list['conv']['lambda']

        self.fc1.W = param_list['fc1']['W']
        self.fc1.b = param_list['fc1']['b']
        self.fc1.params['W'] = self.fc1.W
        self.fc1.params['b'] = self.fc1.b
        self.fc1.weight_decay = param_list['fc1']['weight_decay']
        self.fc1.weight_decay_lambda = param_list['fc1']['lambda']

        self.fc2.W = param_list['fc2']['W']
        self.fc2.b = param_list['fc2']['b']
        self.fc2.params['W'] = self.fc2.W
        self.fc2.params['b'] = self.fc2.b
        self.fc2.weight_decay = param_list['fc2']['weight_decay']
        self.fc2.weight_decay_lambda = param_list['fc2']['lambda']
        
    def save_model(self, save_path):
        param_list = {
            'config': {
                'in_channels': self.in_channels,
                'conv_channels': self.conv_channels,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'padding': self.padding,
                'num_classes': self.num_classes,
                'input_hw': self.input_hw,
                'hidden_dim': self.hidden_dim,
                'dropout_rate': self.dropout_rate,
                'weight_decay': self.conv.weight_decay,
                'weight_decay_lambda': self.conv.weight_decay_lambda,
            },
            'conv': {
                'W': self.conv.params['W'],
                'b': self.conv.params['b'],
                'weight_decay': self.conv.weight_decay,
                'lambda': self.conv.weight_decay_lambda,
            },
            'fc1': {
                'W': self.fc1.params['W'],
                'b': self.fc1.params['b'],
                'weight_decay': self.fc1.weight_decay,
                'lambda': self.fc1.weight_decay_lambda,
            },
            'fc2': {
                'W': self.fc2.params['W'],
                'b': self.fc2.params['b'],
                'weight_decay': self.fc2.weight_decay,
                'lambda': self.fc2.weight_decay_lambda,
            },
        }

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN_Deep(Layer):
    """
    A slightly deeper CNN: Conv-ReLU-Conv-ReLU-MLP head.
    """
    def __init__(
        self,
        in_channels=1,
        conv1_channels=8,
        conv2_channels=16,
        kernel_size=3,
        stride1=1,
        stride2=2,
        padding=1,
        num_classes=10,
        input_hw=(28, 28),
        hidden_dim=128,
        dropout_rate=0.1,
        weight_decay=False,
        weight_decay_lambda=1e-4,
    ):
        self.in_channels = in_channels
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.padding = padding
        self.num_classes = num_classes
        self.input_hw = input_hw
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        h1 = (input_hw[0] + 2 * padding - kernel_size) // stride1 + 1
        w1 = (input_hw[1] + 2 * padding - kernel_size) // stride1 + 1
        h2 = (h1 + 2 * padding - kernel_size) // stride2 + 1
        w2 = (w1 + 2 * padding - kernel_size) // stride2 + 1
        self.conv1_output_shape = (conv1_channels, h1, w1)
        self.conv2_output_shape = (conv2_channels, h2, w2)
        self.flatten_dim = conv2_channels * h2 * w2

        self.conv1 = conv2D(
            in_channels=in_channels,
            out_channels=conv1_channels,
            kernel_size=kernel_size,
            stride=stride1,
            padding=padding,
            weight_decay=weight_decay,
            weight_decay_lambda=weight_decay_lambda,
        )
        self.relu1 = ReLU()
        self.conv2 = conv2D(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=kernel_size,
            stride=stride2,
            padding=padding,
            weight_decay=weight_decay,
            weight_decay_lambda=weight_decay_lambda,
        )
        self.relu2 = ReLU()
        self.fc1 = Linear(
            in_dim=self.flatten_dim,
            out_dim=hidden_dim,
            weight_decay=weight_decay,
            weight_decay_lambda=weight_decay_lambda,
        )
        self.relu_fc = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Linear(
            in_dim=hidden_dim,
            out_dim=num_classes,
            weight_decay=weight_decay,
            weight_decay_lambda=weight_decay_lambda,
        )
        self.layers = [
            self.conv1,
            self.relu1,
            self.conv2,
            self.relu2,
            self.fc1,
            self.relu_fc,
            self.dropout,
            self.fc2,
        ]
        self._conv2_activated_shape = None

    def __call__(self, X):
        return self.forward(X)

    def set_training(self, training=True):
        self.dropout.set_training(training)

    def forward(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], self.in_channels, self.input_hw[0], self.input_hw[1])
        elif X.ndim == 3:
            X = X.reshape(X.shape[0], self.in_channels, X.shape[1], X.shape[2])
        outputs = self.conv1(X)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu2(outputs)
        self._conv2_activated_shape = outputs.shape
        outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = self.fc1(outputs)
        outputs = self.relu_fc(outputs)
        outputs = self.dropout(outputs)
        return self.fc2(outputs)

    def backward(self, loss_grad):
        grads = self.fc2.backward(loss_grad)
        grads = self.dropout.backward(grads)
        grads = self.relu_fc.backward(grads)
        grads = self.fc1.backward(grads)
        grads = grads.reshape(self._conv2_activated_shape)
        grads = self.relu2.backward(grads)
        grads = self.conv2.backward(grads)
        grads = self.relu1.backward(grads)
        return self.conv1.backward(grads)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        config = param_list['config']
        self.__init__(**config)

        for name in ['conv1', 'conv2', 'fc1', 'fc2']:
            layer = getattr(self, name)
            layer.W = param_list[name]['W']
            layer.b = param_list[name]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[name]['weight_decay']
            layer.weight_decay_lambda = param_list[name]['lambda']

    def save_model(self, save_path):
        param_list = {
            'config': {
                'in_channels': self.in_channels,
                'conv1_channels': self.conv1_channels,
                'conv2_channels': self.conv2_channels,
                'kernel_size': self.kernel_size,
                'stride1': self.stride1,
                'stride2': self.stride2,
                'padding': self.padding,
                'num_classes': self.num_classes,
                'input_hw': self.input_hw,
                'hidden_dim': self.hidden_dim,
                'dropout_rate': self.dropout_rate,
                'weight_decay': self.conv1.weight_decay,
                'weight_decay_lambda': self.conv1.weight_decay_lambda,
            },
        }
        for name in ['conv1', 'conv2', 'fc1', 'fc2']:
            layer = getattr(self, name)
            param_list[name] = {
                'W': layer.params['W'],
                'b': layer.params['b'],
                'weight_decay': layer.weight_decay,
                'lambda': layer.weight_decay_lambda,
            }

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
