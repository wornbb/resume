import torch
from torch.nn.parameter import Parameter
from torch.nn import init
class AutoEncoder(torch.nn.Module):
    def __init__(self, layers_config):
        super().__init__()
        self.weights = torch.nn.ParameterList()
        self.biases = torch.nn.ParameterList()
        index = 0
        while index < len(layers_config) - 1:
            self.weights.append(Parameter(torch.Tensor(layers_config[index+1], layers_config[index])))
            self.biases.append(Parameter(torch.Tensor(layers_config[index+1])))
            index += 1
        self.reset_parameters()
    def reset_parameters(self):
        for weight in self.weights:
            init.uniform_(weight,a=0.1)
        for bias in self.biases:
            init.uniform_(bias, a=0.1)
    def forward(self, input):
        output = input
        for weight, bias in zip(self.weights, self.biases):
            output = output.matmul(weight.t()) + bias
            output = torch.nn.functional.tanh(output)
        return output
if __name__ == "__main__":
    N = 10000
    n = 1000
    train = torch.randn(N, 128)
    test = torch.randn(n, 128)
    myEncoder = AutoEncoder([128, 64, 32, 64, 128])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(myEncoder.parameters(), lr=1e-5)
    for t in range(500):
        for i in range(N):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = myEncoder(train[i,:])

            # Compute and print loss
            loss = criterion(y_pred, train[i,:])
            if t % 100 == 99:
                y = myEncoder(test)
                new_loss = criterion(y, test)
                print(t, new_loss.item())
                print(t, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    