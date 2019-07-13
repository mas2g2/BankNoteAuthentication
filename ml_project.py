import numpy as np
import csv

def load_data(filename):
    with open(filename,'r') as f:
        data = list(csv.reader(f))
    data = np.array(data).astype(float)
    return data
class LogisticRegression():

    def __init__(self):
        np.random.seed(1)
        self.theta = np.random.rand(4,1)
    
    def sigmoid(self,x,deriv=False):
        if deriv == False:
            return 1/(np.exp(-x) + 1)
        return x*(1-x)

    def g_x(self,x):
        return x.dot(self.theta)

    def predict(self,x):
        pred = self.sigmoid(self.g_x(x))
        for i in range(len(x)):
            if pred[i] <= 0.5:
                pred[i] = 0
            else:
                pred[i] = 1
        return pred

    def cost_function(self,x,y):
        cost,pred = np.zeros((len(y),1)),self.predict(x)
        for i in range(len(x)):
            cost[i] = y[i]*np.log(pred[i]) - (1-y[i])*np.log(1-pred[i])
        return -np.mean(cost)
   
    def gradient_desc(self,x,y,lr):
       # Gets sample size
       N = len(x)
       # Calculate predictions
       predictions = self.predict(x)
       predictions = np.squeeze(predictions)
       # Gradient desc
       gradient = np.dot(x.T,np.subtract(predictions,y))

       gradient /= N

       gradient *= lr
       gradient = np.reshape(gradient,(4,1))
       self.theta -= gradient

    def fit(self,x,y,lr,iters=1000):
        cost= []
        for i in range(iters):
            err = self.cost_function(x,y)
            print("Error: ",err)
            self.gradient_desc(x,y,lr)

    def score(self,y,pred):
        err_count = 0
        for i in range(len(y)):
            if pred[i] != y[i]:
                err_count += 1
        return 1 - err_count/len(y)

data = load_data("data_banknote_authentication.txt")
train_0,test_0 = data[:661,:],data[662:762,:]
train_1,test_1 = data[762:1271,:],data[1272:,:]
train = train_0
train = np.append(train,train_1,axis=0)
test = np.append(test_0,test_1,axis=0)
print(train.shape)
print(test.shape)
train_x,train_y,test_x,test_y = train[:,:4],train[:,4],test[:,:4],test[:,4]
model = LogisticRegression()
print(model.theta)
model.fit(train_x,train_y,0.001)
print(model.score(test_y,model.predict(test_x)))
