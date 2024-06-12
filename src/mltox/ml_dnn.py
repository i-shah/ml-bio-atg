from torch import nn, optim
from skorch import NeuralNetClassifier, NeuralNetRegressor
import skorch
## PyTorch Descriptor-Based Model approach.
class DNN(nn.Module):
    def __init__(self,num_units=[64,128,256],nonlin=nn.ReLU(),dropout=0.0,input_dim=2000):
        super(DNN, self).__init__()

        self.dense0 = nn.Linear(input_dim, num_units[0])
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units[0], num_units[1])    
        self.dense2 = nn.Linear(num_units[1], num_units[2]) 
        # self.dense3 = nn.Linear(num_units[2], num_units[3])        
        self.output = nn.Linear(num_units[2], 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense2(X))
        # X = self.dropout(X)
        # X = self.nonlin(self.dense3(X))
        X = self.softmax(self.output(X))
        return X

class InputShapeSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        net.set_params(module__input_dim=X.shape[1])

def Classifier_DNN(max_epochs=10,batch_size=128,lr=0.01,device='cpu'):
    net = NeuralNetClassifier(
                            DNN,
                            max_epochs=max_epochs,
                            batch_size=batch_size,
                            lr=lr,
                            optimizer=optim.Adadelta,
                            train_split=False,
                            callbacks =[InputShapeSetter()],
                            verbose=0,
                            device=device
    )
    return net
    
def Regressor_DNN(max_epochs=10,batch_size=128,lr=0.01,device='cpu'):
    net = NeuralNetClassifier(
                        DNN,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        lr=lr,
                        optimizer=optim.Adadelta,
                        train_split=False,
                        callbacks =[InputShapeSetter()],
                        verbose=0,
                        device=device
                        )
    return net