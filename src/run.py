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
expanded_training_data, _, _ = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
mini_batch_size = 10
net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2)),
    FullyConnectedLayer(n_in=40*4*4, n_out=1000, p_dropout=0.5),
    FullyConnectedLayer(n_in=1000, n_out=1000, p_dropout=0.5),
    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], mini_batch_size)
net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.0)