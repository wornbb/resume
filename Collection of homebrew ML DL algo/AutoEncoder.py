import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
from myLayers import FCLayer
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        features = [input_size] + hidden_layers + [input_size]
        for index in range(len(features) - 1):
            self.layers.append(FCLayer(features[index], features[index + 1]))

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer(output)
        return output
if __name__ == "__main__":
    N = 5000
    n = 100000
    train = torch.randint(low=0, high=1000, size=(N, 128), dtype=torch.float)
    test = torch.randint(low=0, high=1000, size=(n, 128), dtype=torch.float)
    myEncoder = AutoEncoder(128, [256, 8, 256])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(myEncoder.parameters(), lr=1e-5)
    for t in range(500):
        for i in range(N):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = myEncoder(train[i,:])
            # Compute and print loss
            loss = criterion(y_pred, train[i,:])
            if i % 1000 == 0:
                cost = torch.sum(y_pred * train[i,:]) / (torch.sqrt(torch.sum(y_pred ** 2)) * torch.sqrt(torch.sum(train[i,:] ** 2)))
                print(t, loss.item())
                print(t, cost)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    