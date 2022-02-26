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
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        self.num_layers = numOfLayers
        self.num_neurons = numOfNeurons # this is a vector of neurons per layer
        self.num_inputs = inputSize
        self.activation = activation # again, a vector of activations for each layer
        self.loss = loss
        self.lr = lr

        # for graphing purposes
        self.losses = []
        
        if type(weights) != type(None):
            self.weights = weights
        else:
            self.weights = [None for i in range(self.num_layers)]

        # create list of layers to hold on to
        self.all_layers = []
        for i in range(self.num_layers):
            if i == 0:
                layer = FullyConnected(self.num_neurons[i], self.activation[i], self.num_inputs, self.lr, self.weights[i])
            else:
                layer = FullyConnected(self.num_neurons[i], self.activation[i], self.num_neurons[i-1], self.lr, self.weights[i])

            self.all_layers.append(layer)
    
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

if __name__=="__main__":
    if (len(sys.argv)<3):
        print('Usage: python main.py [learning_rate (type:float)] ["example" / "and" / "xor"]')
        exit()

    lr = sys.argv[1]
        
    if (sys.argv[2]=='example'):
        print('example from class (single step)')

        # initialize the network
        f = NeuralNetwork(2, np.array(([2, 2])), 2, [1, 1, 1], 0, 0.5, np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]]))
        
        # make the initial prediction and get error
        initial_prediction = f.calculate([0.05,0.1])
        print(f'\nInitial Prediction: {initial_prediction}')
        print(f'Initial Error:      {f.calculateloss(initial_prediction, [0.01, 0.99])}')
        
        # train it for a single step
        f.train([0.05, 0.1], [0.01, 0.99])
        
        # predict again after training
        next_prediction = f.calculate([0.05,0.1])
        print(f'\nNext Prediction:  {next_prediction}')
        print(f'Next Error:       {f.calculateloss(next_prediction, [0.01, 0.99])}')

        # print the new weights
        print('\n\nNEW WEIGHTS AFTER 1 TRAINING STEP')
        w = 1
        b = 1
        for layer in f.all_layers:
            for neuron in layer.all_neurons:
                for i in range(len(neuron.weights)):
                    if i == len(neuron.weights)-1:
                        print(f'b{b}: {neuron.weights[i]}')
                        b += 1
                    else:
                        print(f'w{w}: {neuron.weights[i]}')
                        w += 1
                print()
        
    elif(sys.argv[2]=='and'):
        print('learn and')
        # initialize the network
        f = NeuralNetwork(1, np.array([1]), 2, [1], 1, float(lr))

        print('\n\nAND MODEL USING SIGMOID ACTIVATION AND BINARY CROSS ENTROPY')
        # train for 1000 epochs
        for i in range(10000):
            f.train([0, 0], [0])
            f.train([1, 0], [0])
            f.train([1, 1], [1])
            f.train([0, 1], [0])

        # print the results
        o1 = f.calculate([0, 0])[0]
        o2 = f.calculate([0, 1])[0]
        o3 = f.calculate([1, 0])[0]
        o4 = f.calculate([1, 1])[0]
        print(f'\nWITHOUT STEP FUNCTION')
        print(f'Prediction for [0,0]: {o1}')
        print(f'Prediction for [0,1]: {o2}')
        print(f'Prediction for [1,0]: {o3}')
        print(f'Prediction for [1,1]: {o4}')

        print(f'\nWITH STEP FUNCTION OF >=0.5 = 1, <0.5 = 0')
        print(f'Prediction for [0,0]: {int(o1>=0.5)}')
        print(f'Prediction for [0,1]: {int(o2>=0.5)}')
        print(f'Prediction for [1,0]: {int(o3>=0.5)}')
        print(f'Prediction for [1,1]: {int(o4>=0.5)}')

        print('\n\nAND MODEL USING LINEAR ACTIVATION AND MSE')
        # initialize the network
        f = NeuralNetwork(1, np.array([1]), 2, [0], 0, float(lr))

        # train for 1000 epochs
        for i in range(10000):
            f.train([0, 0], [0])
            f.train([1, 0], [0])
            f.train([1, 1], [1])
            f.train([0, 1], [0])

        # print the results
        o1 = f.calculate([0, 0])[0]
        o2 = f.calculate([0, 1])[0]
        o3 = f.calculate([1, 0])[0]
        o4 = f.calculate([1, 1])[0]
        print(f'\nWITHOUT STEP FUNCTION')
        print(f'Prediction for [0,0]: {o1}')
        print(f'Prediction for [0,1]: {o2}')
        print(f'Prediction for [1,0]: {o3}')
        print(f'Prediction for [1,1]: {o4}')

        print(f'\nWITH STEP FUNCTION OF >=0.5 = 1, <0.5 = 0')
        print(f'Prediction for [0,0]: {int(o1>=0.5)}')
        print(f'Prediction for [0,1]: {int(o2>=0.5)}')
        print(f'Prediction for [1,0]: {int(o3>=0.5)}')
        print(f'Prediction for [1,1]: {int(o4>=0.5)}')

        # graphing code
        # learning_rates = [0.001, 0.01, 0.1, 1]
        # for eta in learning_rates:
        #     f = NeuralNetwork(1, np.array([1]), 2, [0], 0, eta, np.array([[[0.5,0.5,0.5]]]))
        #     for i in range(1000):
        #         f.train([0, 0], [0])
        #         f.train([1, 0], [0])
        #         f.train([1, 1], [1])
        #         f.train([0, 1], [0])

            
        #     print(f'\n\n\n\nETA = {eta}')
        #     print(f.calculate([0, 0]))
        #     print(f.calculate([1, 0]))
        #     print(f.calculate([1, 1]))
        #     print(f.calculate([0, 1]))

        #     temp_losses = []
        #     epoch = []
        #     for i in range(4000):
        #         if i % 100 == 0:
        #             temp_losses.append(f.losses[i])
        #             epoch.append(i)
            
        #     plt.plot(epoch, temp_losses, label=f'lr: {eta}')
        #     #plt.plot(f.losses, label=f'lr: {eta}')
            
        # leg = plt.legend()
        # plt.title('Loss Through Epochs')
        # plt.ylabel('Loss')
        # plt.xlabel('Epochs')
        # plt.savefig('test.png')
        # print('\n\n\n\n')
        # print(f.calculate([0, 0]))
        # print(f.calculate([1, 0]))
        # print(f.calculate([1, 1]))
        # print(f.calculate([0, 1]))
        
    elif(sys.argv[2]=='xor'):
        print('learn xor')
        print('XOR as a perceptron\n')
        f = NeuralNetwork(1, np.array([1]), 2, [1], 1, float(lr))
        for i in range(10000):
            f.train([0, 0], [0])
            f.train([1, 0], [1])
            f.train([1, 1], [0])
            f.train([0, 1], [1])

        # print the results
        o1 = f.calculate([0, 0])[0]
        o2 = f.calculate([0, 1])[0]
        o3 = f.calculate([1, 0])[0]
        o4 = f.calculate([1, 1])[0]
        print(f'WITHOUT STEP FUNCTION')
        print(f'Prediction for [0,0]: {o1}')
        print(f'Prediction for [0,1]: {o2}')
        print(f'Prediction for [1,0]: {o3}')
        print(f'Prediction for [1,1]: {o4}')

        print(f'\nWITH STEP FUNCTION OF >=0.5 = 1, <0.5 = 0')
        print(f'Prediction for [0,0]: {int(o1>=0.5)}')
        print(f'Prediction for [0,1]: {int(o2>=0.5)}')
        print(f'Prediction for [1,0]: {int(o3>=0.5)}')
        print(f'Prediction for [1,1]: {int(o4>=0.5)}')

        # graphing code
        # learning_rates = [0.001, 0.01, 0.1, 1, 10]
        # for eta in learning_rates:
        #     f = NeuralNetwork(1, np.array([1]), 2, [1], 1, eta, np.array([[[0.1, 0.2, 0.3]]]))
        #     for i in range(25):
        #         f.train([0, 0], [0])
        #         f.train([1, 0], [1])
        #         f.train([1, 1], [0])
        #         f.train([0, 1], [1])

        #     print(f'\n\n\n\nETA = {eta}')
        #     print(f.calculate([0, 0]))
        #     print(f.calculate([1, 0]))
        #     print(f.calculate([1, 1]))
        #     print(f.calculate([0, 1]))
            
        #     temp_losses = []
        #     epoch = []
        #     for i in range(100):
        #         if i % 1 == 0:
        #             temp_losses.append(f.losses[i])
        #             epoch.append(i)
            
        #     plt.plot(epoch, temp_losses, label=f'lr: {eta}')

        # leg = plt.legend(loc='upper right')
        # plt.title('Loss Through Epochs')
        # plt.ylabel('Loss')
        # plt.xlabel('Epochs')
        # plt.savefig('test.png')
        # print('\n\n\n\n')
        # print(f.calculate([0, 0]))
        # print(f.calculate([1, 0]))
        # print(f.calculate([1, 1]))
        # print(f.calculate([0, 1]))
        
        print('\n\nXOR with a single hidden layer')
        print('NOTE: Please use a learning rate between 0.1 and 1 for best results')
        print('      sometimes, local minima may cause model to get stuck. Please retry with weight between 0.1 and 1')
        f = NeuralNetwork(2, np.array([8, 1]), 2, [1, 1], 1, float(lr))
        for i in range(10000):
            f.train([0, 0], [0])
            f.train([1, 0], [1])
            f.train([1, 1], [0])
            f.train([0, 1], [1])

        # print the results
        o1 = f.calculate([0, 0])[0]
        o2 = f.calculate([0, 1])[0]
        o3 = f.calculate([1, 0])[0]
        o4 = f.calculate([1, 1])[0]
        print(f'\nWITHOUT STEP FUNCTION')
        print(f'Prediction for [0,0]: {o1}')
        print(f'Prediction for [0,1]: {o2}')
        print(f'Prediction for [1,0]: {o3}')
        print(f'Prediction for [1,1]: {o4}')

        print(f'\nWITH STEP FUNCTION OF >=0.5 = 1, <0.5 = 0')
        print(f'Prediction for [0,0]: {int(o1>=0.5)}')
        print(f'Prediction for [0,1]: {int(o2>=0.5)}')
        print(f'Prediction for [1,0]: {int(o3>=0.5)}')
        print(f'Prediction for [1,1]: {int(o4>=0.5)}')

        # graphing code
        # learning_rates = [0.001, 0.01, 0.1, 1]
        # for eta in learning_rates:
        #     f = NeuralNetwork(2, np.array([2, 1]), 2, [1, 1], 1, eta)
        #     for i in range(1000):
        #         f.train([0, 0], [0])
        #         f.train([1, 0], [1])
        #         f.train([1, 1], [0])
        #         f.train([0, 1], [1])

        #     print(f'\n\n\n\nETA = {eta}')
        #     print(f.calculate([0, 0]))
        #     print(f.calculate([1, 0]))
        #     print(f.calculate([1, 1]))
        #     print(f.calculate([0, 1]))
            
        #     temp_losses = []
        #     epoch = []
        #     for i in range(4000):
        #         if i % 50 == 0:
        #             temp_losses.append(f.losses[i])
        #             epoch.append(i)
            
        #     plt.plot(epoch, temp_losses, label=f'lr: {eta}')

        # leg = plt.legend()
        # plt.title('Loss Through Epochs')
        # plt.ylabel('Loss')
        # plt.xlabel('Epochs')
        # plt.savefig('test.png')
        # print('\n\n\n\n')
        # print(f.calculate([0, 0]))
        # print(f.calculate([1, 0]))
        # print(f.calculate([1, 1]))
        # print(f.calculate([0, 1]))