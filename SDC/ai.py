#BRAIN OF THE CAR

import numpy as np
import random
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
## Architecture of neural network
#             Inheriting from Module class
# class used to define and connect various layers(input, hidden and output) and form the architecture of the network
class Network(nn.Module):
                                    #nb_action is the output size
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__() ## Calling the constructor of the parent class(Module class of nn module)
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 40) #full connection between input layer and hidden layers(we do not make layers explicitly and make full connections, there are 30 neuron in hidden layer and only one hidden layer)
        self.fc2 = nn.Linear(40, nb_action) #full connection between hidden layer and output layer
        ## Output for each neuron will be linear combination of previous inputs

    #function to forward propagate the inputs and get outputs
                        #state will be the 5 inputs in tensor format
    def forward(self, state):
        x = F.relu(self.fc1(state))  # we will get the outputs from the first full connection and apply relu(activation function) to the outputs
        q_values = self.fc2(x) # outputs obtained from hidden layer after applying relu are passed as inputs to fc2 and gives final outputs to output layer
        # we will have q values as final output and will apply softmax activation to them later
        #input->linear combination -> relu ->linear combination ->output
        return q_values

#Experience replay to prevent the model from learning a particular sequentially correlated input and to provide rare experience again

class ReplayMemory():
                        #capacity of memory used for replay
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] #initializing memory as an empty list (will store events)
        #       event will be a 4 tuple = (last_state,new_state,last_action,last_reward)
        #  last_state is the state just before taking last_action
        #  after taking last_action we will land up in new_state
        # last_action is latest action we have taken
        #  last_reward is the reward that we get after we take the last_action and land up in the new_state
        # each state is 5 tuples of input
        # to add the latest experience in the memory
    def push(self,event):
        self.memory.append(event)
        # all 4 attributes of the event(last_state, next_state, last_action, last_reward) would be tensors
        # if memory exceeds the capacity we delete the oldest experience
        if len(self.memory)>self.capacity:
            del self.memory[0]


    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        # random.sample(self.memory, batch_size) gives a random sample having batch_size no of elements from the self.memory
        # zip(*) is used for reshaping
        #ex zip(*[[1,2,3],[4,5,6]]) = [1,4],[2,5],[3,6]
        # after applying reshaping we would be able to separate our inputs in the form list of last_states list of new_states...
        # this format is required for pytorch
        # we will now wrap these batches in a pytorch variable(a pytorch variable contains both a tensor and its associated gradient)

        # we will be able to get an associated gradient for each particular type of parameter(last_state,last_reward etc.) for given batch
        # pytorch_variable =
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

class Dqn():
                                              #gamma is learning rate
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        # model is required as name of attribute
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) # we want 100000 events inside our replay memory
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001) #optimizer function while doing backpropagation
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # 0 is the index of the new dimension we will add
                            #torch.Tensor (input_size) makes a tensor of 1 dimension and 5 elements and we add another dimension corresponding to the batch using unsqueeze(0)
        # last_state is a tuple of 5 elements(hence 1 dimension), however for pytorch we need it as a torch tensor with 1 extra dimension for batch
        self.last_action = 0 #action are 0,1,2
        self.last_reward = 0

                            #state will already be as a tensor
    # function to get output on a particular input state (assuming that model is already trained)
    def select_action(self, state):
        q_values = self.model.forward(Variable(state, volatile = True)) # will wrap the state tensor into a pytorch variable so that we get the associated gradient
        #                                               however this gradient would not be required as this data is not used for training we set volatile = true(we don't need to do backpropagation)
        Temperature = 150 # higher the temperature parameter more sure the model will be while taking its action(Higher temperature, higher will be the probability associated with the maximum q_value)
        probabilities = F.softmax(q_values*Temperature)
        # we need to take a random draw from the given probabilities to get the final action
        action = probabilities.multinomial(num_samples=1) # multinomial gives a random draw from the given distribution
        # action will be a pytorch_variable(tensor+fake dimension for batch)
        return action.data[0,0] #getting the exact action from the tensor

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # batch_action = batch_action.type(torch.int64)
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)#kill the fake batch dimension with squeeze
        # self.model.forward(batch_state) gives us the q_values for each of the input state and all 3 actions for a given input state, however we want q value for only one of the action for a given input state and this particular action is the one present in the batch_actions
        # this would give the Q values for all elements of the batch as Q(stb,atb)
        # we don't need to add any conversion of batch_state as it is already in batch format and will be a pytorch object
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # detach all the columns and select the one having maximum q_value
        # we need to get maximum(over all actions) of all q values for a given next_state
        target = batch_reward+self.gamma*next_outputs
        td_loss = F.smooth_l1_loss(outputs, target) ##computing the loss function that we need to minimise
        self.optimizer.zero_grad() # initialise the gradient to zero at each iteration of the loop
        td_loss.backward(retain_graph=True) #backpropagating the error (retain graph=True will mean that you will retain your vertices in your graph which is used while backpropagation, if we remove this then it would lead to errors)
        self.optimizer.step() # update the weights depending on the back propagated error

    # update function is to update the parameters(last_state,last_reward etc.), replay memory, and then train the neural network again after each step ta
    # update will be called after each step(change of state)
    def update(self, reward, newState):
                            #newState is just a tuple of 5 elements that we convert into tensor and add a batch dimension
        new_state = torch.Tensor(newState).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward])))
                                                        # we first convert the given last_action and last_reward to tensor
        action = self.select_action(new_state)
        if len(self.memory.memory)>100: # we won't start learning until we get at least 100 events
                # first memory refers to the object of ReplayMemory class second memory is the list attribute of the object
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) # take a sample of batch_size 100 from memory and unpack it
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_reward = reward
        self.last_state = new_state
        self.reward_window.append(reward)

        if len(self.reward_window)>1000:
            del self.reward_window[0]
        return action

    def score(self): # the score will be the latest mean that we need to add in the sliding window
        return sum(self.reward_window)/(len(self.reward_window)+0.2) #1 added to avoid error as initially your reward window would be empty

    def save(self):
        torch.save({
            'state_dict':self.model.state_dict(),  # attributes of the model (weights and bias )
            'optimizer': self.optimizer.state_dict()}, # attributes of optimizer
            'last_brain.pth') # name of file where we want to save the attributes

    def load(self):
        if os.path.isfile('last_brain.pth'): # if there exists saved weights then we load the weights
            checkpoint = torch.load('last_brain.pth') # checkpoint will be a dictionary that we saved
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
