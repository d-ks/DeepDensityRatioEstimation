# Density ratio estimation by DeepLogReg
# proposed by Uchibe[2018]
# code by D.Kishikawa

import numpy as np

import os
import sys

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, Chain, serializers

###### DEFINE NETWORK ######

class Network(Chain):

    # Multi-Layered Perceptron, 4 layer.
    ## n_input : num of neuron in Input layer.
    ## n_hidden : num of neuron in Hidden layer.
    ## n_output : num of neuron in Output layer.

    def __init__(self,n_input, n_hidden, n_output):
        super(Network, self).__init__()
        initW = chainer.initializers.HeNormal()
        with self.init_scope():
            self.layer1 = L.Linear(n_input, n_hidden, initialW=initW)
            self.layer2 = L.Linear(n_hidden, n_hidden, initialW=initW)
            self.layer3 = L.Linear(n_hidden, n_output, initialW=initW)

            self.train = True

    def __call__(self, x):
        h1 = F.relu(self.layer1(x))
        h2 = F.dropout(F.relu(self.layer2(h1)), ratio=0.5)
        h3 = self.layer3(h2)
        return h3

    def compute(self, x):
        h1 = F.relu(self.layer1(x))
        h2 = F.relu(self.layer2(h1))
        h3 = self.layer3(h2)
        return h3


###### LOSS FUNCTION ######

def compute_f_loss(n_e, n_b, f_exp, f_base):
    # Loss function of f-network
    ## Note: The final term (L2norm) of loss fun is called "weight decay",
    ## so not implemented in here.

    # 1st term: f_base.
    L_base = - (1 / n_b) * F.sum(F.log(1 - F.sigmoid( f_base ) + 0.1**10))
    # 2nd term: f_exp.
    L_exp = - (1 / n_e) * F.sum(F.log(F.sigmoid( f_exp ) + 0.1**10))

    f_loss = L_base + L_exp

    return f_loss

#Set seed of random num.
np.random.seed(0)

### 1. training f_network
print("=== TRAIN ===")

print("checking existence of directory to save model...")

if os.path.isdir("./model_LogReg/")!=True:
    os.mkdir("./model_LogReg/")
    print("  => not exist. made directory.")
else:
    print("  => exist.")


print("Loading data for training f...")

if os.path.isfile("./data/exp.csv")!=True or os.path.isfile("./data/exp.csv")!=True or os.path.isdir("./data/")!=True:
    print("  ==> no dataset! aborted.")
    sys.exit()
else:
    # load data (for f)
    s_exp_data_f = np.loadtxt("./data/exp.csv").astype(np.float32)
    s_base_data_f = np.loadtxt("./data/base.csv").astype(np.float32)

print("  ==> done.")

# adjust number of data
n_exp = np.shape(s_exp_data_f)[0]
n_base = np.shape(s_base_data_f)[0]

n_state = 1


if n_exp > n_base:
    n_data = n_base
else:
    n_data = n_exp

# convert to chainer.Variable
s_exp = chainer.Variable(s_exp_data_f.reshape((n_data,1)))
s_base = chainer.Variable(s_base_data_f.reshape((n_data,1)))

print("building network...")

# build network
n_input = 1
n_hidden = 100
n_output = 1

f_network = Network(n_input, n_hidden, 1)

# set optimizer SGD
optimizer = optimizers.SGD().setup(f_network)
#weight decay = the last term of Loss function
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

print("  ==> done.")

# set epoch, step, batchsize
n_epoch = 1000
n_step = 10
batch_size = 10

#note: these are NOT hyperparameters.
best_epoch = 0
old_loss = 100

print("=== START TRAINING ===")

for epoch in range(n_epoch):
    loss_list = []
    lossv = []

    for _ in range(n_step):

        i = np.random.randint(0, n_data - 1 - batch_size)

        #Clear gradients.
        f_network.cleargrads()

        #compute f_exp
        s_exp_data = s_exp[i : i+batch_size]
        f_exp = f_network(s_exp_data)

        #compute f_base
        s_base_data = s_base[i : i+batch_size]
        f_base = f_network(s_base_data)

        #compute f-loss
        f_loss = compute_f_loss(batch_size, batch_size, f_exp, f_base)
        loss_list.append(f_loss.data)

        #backpropagation of f-loss
        f_loss.backward()

        #Update optimizer
        optimizer.update()

    losse = np.mean(np.array(loss_list))

    if losse < old_loss:
        serializers.save_npz("./model_LogReg/f_net_" + str(epoch) + ".npz", f_network)
        old_loss = losse
        best_epoch = epoch
        print("End epoch: {0} /  loss: {1} -- model saved".format(epoch, losse))
    else:
        print("End epoch: {0} /  loss: {1}".format(epoch, losse))

print("=== END OF TRAINING ===")

#compute density ratio by trained network in range(-5.0~5.0)
print("computing estimated density ratio...")
serializers.load_npz("./model_LogReg/f_net_" + str(best_epoch) + ".npz", f_network)
r_ln_arr = np.zeros((101,2))
for s in range(-50,51):
    f_value = f_network.compute(chainer.Variable(np.array([[s/10]]).astype(np.float32))).data
    r_ln = f_value + np.log(n_base/n_exp)
    r_ln_arr[int(s + 50)][0] = s/10
    r_ln_arr[int(s + 50)][1] = np.exp(r_ln)

print("saving density ratio...")
np.savetxt("r_LogReg.csv",r_ln_arr, delimiter=",")
print("  ==> Estimated density ratio saved in r_LogReg.csv!")