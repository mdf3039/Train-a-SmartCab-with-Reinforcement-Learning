import numpy as np
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.time = 0
        #The Trial Number
        self.tn = 0
        #If the trial is successful, the trial number goes in the list
        self.strials = []
        #The actions the agent can take
        self.actions = (None,'forward','left','right')
        #The number of total successes
        self.success = 0
        #Set up tuple for q values. The dictionary entry will be a string of the state and action str((self.state,action))
        #The value of the dictionary entry will be a the q value
        self.qvalue = {}
        #Count the number of times it goes to each state for learning rate and epsilon. The dictionary entry will be a string
        #of the state. str(self.state). The value will be the number of times the agent has visited the state.
        self.count = {}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.time = 0
        #Add one each time it resets to a new trial
        self.tn += 1

    def update(self, t):      
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        #The state will be the inputs. There is no need for the right input. It does not matter if a car is coming from
        #the right. Also, add next_waypoint to the state. There are 4 tuples: light, oncoming, left, and next_point.
        self.state = inputs
        del self.state["right"]
        self.state["next_point"] = self.next_waypoint
        #Update the time
        self.time += 1
        #Update count for the given state. First, it tries to add one to the state. If this doesn't work, this means the
        #state does not already exist as a tuple in self.count. Therefore this is the first time it is at the state,
        #so it gets a value of 1.
        try:
            self.count[str(self.state)] += 1
        except:
            self.count[str(self.state)] = float(1)
                
        # TODO: Select action according to your policy
        #The count will determine the epsilon. The greedy limit + infinite exploration technique
        #will be used. epsilon will equal 1/count
        epsilon = float(1/self.count[str(self.state)]**2)
        #If the qvalues are not there for the states and actions, make them 0. The only time the count dictionary
        #will equal 1 for a tuple is when it is its first time in that state. Therefore, we set the qvalues equal to 0
        #for all the actions in the state.
        if self.count[str(self.state)] == 1:
            for num in range(0,4):
                self.qvalue[str((self.state,self.actions[num]))] = 0
            
        #Find the action index that has the maximum q value. create a list of all of the qvalues for each action
        #then choose the indices for the maximum. If more than one, randomly choose one.
        #First, an array is created with just the qvalues for the first action in self.actions. Then a while function is
        #called to append the other qvalues for each action into the array. Then it chooses the index where the Qvalue is
        #the largest. If there are multiple with the same values, one index is chosen.
        qvlist = np.array(self.qvalue[str((self.state,self.actions[0]))])
        for num in range(1,4):
            qvlist = np.append(qvlist, self.qvalue[str((self.state,self.actions[num]))])
        #Now that the list of Qvalues is created, find the maximum(s) index denoted by the list m
        m = []
        #For all of the indices of actions, if the maximum of the array equals that index, append the index to the list
        for ind in range(0,4):
            if np.amax(qvlist) == qvlist[ind]:
                m.append(ind)
        #Choose a random index from the list of maximum index(s)
        q_ind = random.choice(m)
        #Now the greedy limit infinite exploration is put into use. A random number between 0 and 1 is chosen. If epsilon
        #is equal to 1 (shouldn't be above 1), then a random action is chosen, since this means this is the first time at the state.
        #If epsilon is smaller than x (it decays so it will usually be smaller), the chosen index gives the correct action.
        #If epsilon is larger than x, a random action is taken.
        x = random.random()
        if epsilon >= 1:
            action = self.actions[random.randrange(1,4)]
        elif x >= epsilon:
            action = self.actions[q_ind]
        elif x < epsilon:
            action = self.actions[random.randrange(1,4)]

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #The count will determine the learning rate(LR). The learning rate will equal 1/count^2
        LR = float(1/self.count[str(self.state)]**2)
        #I need to know what the Qvalue for the next state will be so I can put it in my model.
        #save the old state as old_state and old action as old_action
        old_state = self.state
        old_action = action
        place = self.env.agent_states[self]
        #When the location=destination, the model finishes so the
        #estimate of next state utility (NSU) is 0. Also record success and trial number if success
        if place['location'] == place['destination']:
            NSU = 0
            self.success += 1
            self.strials = np.append(self.strials, self.tn)
        #When location does not equal destination, I must update the state and take the maximum q values of the actions to
        #find the NSU.
        elif place['location'] != place['destination']:
            #Find out where the location of the new agent is with respect to the destination
            # Gather inputs
            self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
            inputs = self.env.sense(self)
            # TODO: Update state. Delete the right input and add the next_waypoint
            self.state = inputs
            del self.state["right"]
            self.state["next_point"] = self.next_waypoint
            #If the qvalues are not there for the new state and actions, make them 0. If I can't add 0 to the count of the
            #state, that means the state does not exist in the count matrix. Therefore, if the state does not exist in the
            #count matrix, the actions for the state do not exist in the qvalue matrix.
            try:
                self.count[str(self.state)] += 0
            except:
                for num in range(0,4):
                    self.qvalue[str((self.state, self.actions[num]))] = 0
            #Find the maximum q value for the next state. Now that the qvalues are there for each action in the new state,
            #we find the maximum Qvalue for the state by making an array of all of the Qvalues and taking the maximum.
            qvlist = np.array(self.qvalue[str((self.state,self.actions[0]))])
            for num in range(1,4):
                qvlist = np.append(qvlist, self.qvalue[str((self.state,self.actions[num]))])
            NSU = np.amax(qvlist)
        #Update the qvalues. Use the old_state and old_action to update the current Q value. alpha is LR or learning rate.
        #reward is reward. Gamma is the number between 0 and 1 that is being multiplied with the next state utility(NSU)
        self.qvalue[str((old_state, old_action))] = (1-LR)*self.qvalue[str((old_state, old_action))] + LR*(reward + 0.5*NSU)
        #I need to know how many agents succeed
        print "Number of successes", self.success
        #I need to know which trial numbers have success
        print "Trial numbers that succeeded", self.strials
        print "Number of times visiting the state", self.count[str(old_state)]
        print "Epsilon", epsilon


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
