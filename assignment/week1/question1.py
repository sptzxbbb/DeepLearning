import math

class Activation:
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    def relu(self, x):
        return max(x, 0)

    def leaky_relu(self, alpha, x):
        if x >= 0:
            return x
        else:
            return alpha * x

    def elu(self, alpha, x):
        if x >= 0:
            return x
        else:
            return alpha * (math.exp(x) - 1)

if __name__ == "__main__":
    activation = Activation()
    print(activation.sigmoid(10))
    print(activation.relu(20))
    print(activation.tanh(-1))
    print(activation.leaky_relu(0.1, -1))
    print(activation.elu(0.1, -1))