import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#import network
#net = network.Network([784, 30, 10])
#net = network.Network([784, 100, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

#import network2
#net = network2.Network([784, 30, 10])
#net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
#net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
#net.SGD(training_data, 30, 10, 0.5, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
#net.SGD(training_data, 30, 10, 0.1, 
#        lmbda=5.0, 
#        evaluation_data=validation_data, 
#        monitor_evaluation_accuracy=True)

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 30
net = Network([
    FullyConnectedLayer(n_in=784, n_out=100),
    SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)