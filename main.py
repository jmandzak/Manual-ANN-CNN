# for graphing purposes only use matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import sys
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        self.activation = activation
        self.num_inputs = input_num + 1 # to account for the bias
        self.lr = lr

        # if weights is passed, use it. If not, just use random list of weights with len = num_inputs
        if type(weights) != type(None):
            self.weights = weights
        else:
            self.weights = [x for x in np.random.randn(self.num_inputs)]
        
    #This method returns the activation of the net
    # linear = 0
    # log = 1
    def activate(self,net):
        # if linear, just return
        if self.activation == 0:
            return net
        else:
            return 1/(1+np.exp(-net))
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        sum = 0
        if len(input) != self.num_inputs:
            input.append(1) # to account for bias
        self.input = np.asarray(input)
        for i, w in zip(input, self.weights):
            sum += i * w
        
        # store the net
        self.net = sum

        # do activation function
        result = self.activate(sum)

        self.output = result
        return result

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.activation == 1:
            return self.output * (1 - self.output)
        else:
            return 1  
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        # multiply the wtimesdelta by the activation function
        # FROM NOTES: Each neuron calculates its own delta by multiplying by the derivative of the activation function
        wtimesdelta_array = np.asarray(wtimesdelta)
        self.delta = wtimesdelta_array * self.activationderivative()

        # FROM NOTES: The neuron returns the vector of 𝑤𝛿to the FullyConnectedLayer
        return self.delta * np.asarray(self.weights)
    
    #Simply update the weights using the partial derivatives and the learning weight
    def updateweight(self):
        # multiply the input by the delta, then update weight
        self.weights = self.weights - (self.lr * self.delta * np.asarray(self.input))

        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        self.num_neurons = numOfNeurons
        self.activation = activation
        self.num_inputs = input_num
        self.lr = lr

        # initialize weights
        if type(weights) != type(None):
            self.weights = weights
        else:
            self.weights = [None for i in range(self.num_neurons)]

        # initialize all the neurons and add them to a list of neurons for the class to keep
        self.all_neurons = []
        for i in range(self.num_neurons):
            n = Neuron(activation, input_num, lr, self.weights[i])
            self.all_neurons.append(n)
        
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        vec = []
        for neuron in self.all_neurons:
            vec.append(neuron.calculate(input))

        return vec
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        all_wtimesdelta = []

        # get all the wtimesdeltas
        for neuron, wdelta in zip(self.all_neurons, wtimesdelta):
            all_wtimesdelta.append(neuron.calcpartialderivative(wdelta))
            # fix the weight for that neuron
            neuron.updateweight()
            
        # sum them and return them
        array = np.array(all_wtimesdelta)
        array = array.sum(axis=0)
        return array
           

# convolutional layer class
class ConvolutionalLayer: 
    def __init__(self, num_kernels, kernel_size, activation, input_dim, lr, weights=None):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_dim = input_dim      # NOTE: this should always be three dimensions, even if the third dimension is 1. Should be a tuple/list of form (width, height, depth)
        self.lr = lr
        self.weights = weights          # should be a 2d vector. first dim = kernel, second dim = weights for that kernel

        # we need to initialize the weights if none were passed
        if self.weights == None:
            self.weights = []   # this will be a vector of vectors, one vector for each kernel
            for i in range(self.num_kernels):
                # number of weights = width of kernel * height of kernel * number of channels + 1 for the bias
                num_weights = self.kernel_size * self.kernel_size * self.input_dim[2] + 1
                kernel_weights = list()
                kernel_weights = [x for x in np.random.randn(num_weights)]
                self.weights.append(kernel_weights)

        # next we need to initialize the kernels
        # because of how the project is set up, instead of treating a kernel like a kernel, we're going to have our "kernels" be
        # a vector of vectors, where each vector is a single kernel, and inside that vector are all of the output neurons since the kernel is really just the weights
        self.all_kernels = []
        # get the number of neurons in a kernel
        # this should be (Wi-Wf + 1) * (Hi - Hf + 1) where W is width, H is height, i is input, f is filter (which is kernel)
        num_neurons = (self.input_dim[0] - self.kernel_size + 1) * (self.input_dim[1] - self.kernel_size + 1)
        
        # save the dimension of the feature map (width only). This is equal to number of slides. Useful for calculation
        self.feature_map_dim = self.input_dim[0] - self.kernel_size + 1

        for i in range(self.num_kernels):
            kernel_neurons = list()
            for j in range(num_neurons):
                # NOTE: input size when initializing the neuron excludes the bias because the neuron adds that itself
                n = Neuron(self.activation, (self.kernel_size ** 2 * self.input_dim[2]), self.lr, self.weights[i])
                kernel_neurons.append(n)

            self.all_kernels.append(kernel_neurons)

    def calculate(self, input):
        # input should be a 2d list. The first dimension is channels, the second dimension is all the inputs for that kernel
        # so a 3x3x2 input would be of form 2x9, for 2 channels and 9 numbers per channel
        
        all_outputs = []    # list of lists, first dimension is kernel, second dimension is feature map from that kernel

        # 6d for loop. try to keep up.
        # first step is each kernel
        for kernel in range(self.num_kernels):
            all_inputs = []
            
            # next up is the number of different starting rows
            for starting_row in range(self.feature_map_dim):
                # next up is the number of starting columns
                for starting_col in range(self.feature_map_dim):

                    # now the following 3d for loop gets all of the inputs for the specific neuron
                    kernel_inputs = []
                    # 3d for loop. First step is all channels
                    for channel in range(self.input_dim[2]):

                        # next step: each row in kernel
                        for row in range(self.kernel_size):

                            # final loop: each column in kernel
                            for col in range(self.kernel_size):
                                actual_row = (starting_row + row) * self.input_dim[0]
                                actual_col = (starting_col + col)
                                kernel_inputs.append(input[channel][actual_row + actual_col])

                    all_inputs.append(kernel_inputs)

            # once we've finished these 5 loops, we have all the inputs we need to find the feature map of this kernel
            feature_map = []
            for neuron, input_val in zip(self.all_kernels[kernel], all_inputs):
                feature_map.append(neuron.calculate(input_val))

            all_outputs.append(feature_map)

        return all_outputs



# Flattening Layer
class FlattenLayer:
    def __init__(self, input_size):
        self.input_size = input_size    # input size should be a tuple or list in the form (num_channels, num_neurons)

    def calculate(self, input):
        output = []
        for channel in range(self.input_size[0]):
            for neuron in range(self.input_size[1]):
                output.append(input[channel][neuron])

        return output


# Max Pooling Layer
class MaxPoolingLayer:
    def __init__(self, kernel_size, input_dim):
        self.kernel_size = kernel_size
        self.input_dim = input_dim      # this should again be a 3d list or tuple (width, height, depth)
        self.feature_map_dim = (self.input_dim[0] - self.kernel_size) / self.kernel_size + 1

    def calculate(self, input):
        self.locations = []     # this will be a 2d list of all locations of max vals. first dim = channel, second dim = locations
        self.max_values = []    # also a 2d matrix, this time of the max vals

        # create list for looping purposes
        stride_loop = [i for i in range(self.input_dim[0]) if i%self.kernel_size == 0]
       

        for channel in range(self.input_dim[2]):
            kernel_max_vals = []
            kernel_max_indices = []
            
            for starting_row in stride_loop:
                # next up is the number of starting columns
                for starting_col in stride_loop:

                    # now the following 3d for loop gets all of the inputs for the specific neuron
                    values = []
                    positions = []

                    # next step: each row in kernel
                    for row in range(self.kernel_size):

                        # final loop: each column in kernel
                        for col in range(self.kernel_size):
                            actual_row = (starting_row + row) * self.input_dim[0]
                            actual_col = (starting_col + col)
                            values.append(input[channel][actual_row + actual_col])
                            positions.append(actual_row + actual_col)
                            
                    max_val = max(values)
                    max_index = values.index(max_val)
                    max_index = positions[max_index]

                    kernel_max_vals.append(max_val)
                    kernel_max_indices.append(max_index)

            
            self.max_values.append(kernel_max_vals)
            self.locations.append(kernel_max_indices)

        return self.max_values 

        


#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, input_size, loss_function, lr):
        self.num_inputs = inputSize
        self.loss = loss
        self.lr = lr

        # for graphing purposes
        self.losses = []
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        for i in range(self.num_layers):
            input = self.all_layers[i].calculate(input)
            
        # return the output
        return input
    
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    # making the assumption that both yp and y are vectors considering MSE we assume multiple output neurons
    # and binary cross entropy we need multiple values to calculate the actual loss
    def calculateloss(self,yp,y):

        # MSE loss
        if self.loss == 0:
            sum = 0
            for i in range(len(yp)):
                val = y[i] - yp[i]
                val = val ** 2
                sum += val

            sum = sum / len(yp)
            return sum

        # otherwise binary cross entropy
        else:
            sum = 0
            for i in range(len(yp)):
                val = -1 * (y[i] * np.log(yp[i]) + (1 - y[i]) * (np.log(1-yp[i])))
                sum += val

            sum = sum / len(yp)
            return sum
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    # this simply does it for one neuron
    def lossderiv(self,yp,y):
        
        # MSE loss
        if self.loss == 0:
            # out - target
            return yp - y
        # Binary Cross Entropy
        else:
            # negative out / target + (1 - target) / (1 - out)
            val = (-1 * y) / yp
            val = val + ((1 - y) / (1 - yp))
            return val
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        # do the forward pass
        input = x
        self.final_output = self.calculate(input)

        # save the loss
        self.losses.append(self.calculateloss(self.final_output, y))

        # now get the last layer's delta
        wtimesdelta = []
        for yp, y_target in zip(self.final_output, y):
            wtimesdelta.append(self.lossderiv(yp, y_target))
        
        # now pass it to each layer
        for layer in reversed(self.all_layers):
            wtimesdelta = layer.calcwdeltas(wtimesdelta)

    # not sure right now what else needs to be put in here but def need more params
    def addLayer(type_layer):
        pass




def main():
    #convo_layer = ConvolutionalLayer(1, 3, 0, (5, 5, 2), 0.1)
    #convo_layer.calculate([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
    #nums = list((x for x in range(25)))
    #convo_layer.calculate([nums, nums])

    convo_layer = ConvolutionalLayer(2, 2, 0, (3,3,2), 0.1, [[1, 0, 0, 1, 0, 1, 1, 0], [2,0, 0,2, 0,2, 2,0]])
    output = convo_layer.calculate([[0, 1, 0, 1, 0, 1, 1, 1, 1], [1,0,1, 0,1,0, 0,0,0]])
    print(output)
    print()

    flatten_layer = FlattenLayer((2, 4))
    output = flatten_layer.calculate(output)
    print(output)

    print('\n')
    m = MaxPoolingLayer(2, (4,4,1))
    output = m.calculate([[0,1,3,2, 2,3,1,0, 4,5,8,9, 7,6,3,4]])
    print(output)


if __name__ == '__main__':
    main()
