# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for idx in range(self.iterations):
            next_values = self.values.copy()
            # print('-----BEGIN ITER: {0}-----'.format(i))
            for state in self.mdp.getStates():
                # print('Current state: {0} (isTerminal={1})'.format(state, self.mdp.isTerminal(state)))
                actions_from_state = self.mdp.getPossibleActions(state)
                max_q_value = -float('inf')
                if (self.mdp.isTerminal(state) or (not len(actions_from_state) > 0)):
                    # print('At terminal state or no actions from state, max_q_value is 0')
                    max_q_value = 0
                else:
                    for action in actions_from_state:
                        q_value = self.computeQValueFromValues(state, action)
                        # print('q_value:{0}\ncurrent max_q_value:{1}'.format(q_value, max_q_value))
                        if q_value > max_q_value:
                            max_q_value = q_value
                            # print('max_q_value is updated')
                    # print('\n\n')
                next_values[state] = max_q_value
            # Update backup (self.values) with next values computed from static full backup
            self.values = next_values
            # print('-----END ITER {0}-----'.format(i))

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # A terminal state has a q-value of 0 (there are no actions to be taken)
        # if self.mdp.isTerminal(state):
        #     return 0

        # This may or may not be correct, does this get used in runValueIteration or is it seperate? 
        next_states_and_probs_list = self.mdp.getTransitionStatesAndProbs(state, action)

        # The q_value is 0 if there are no next states (this covers both the terminal case and if there are no actions in a state, still will check if terminal initially so no other calls to self.mdp have to be made)
        q_value = 0

        # Sum over all s' fro a given s and a
        for next_state, next_state_prob in next_states_and_probs_list:
            q_value += next_state_prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.getValue(next_state))

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        actions_from_state = self.mdp.getPossibleActions(state)
        max_q_value = -float('inf')
        best_action = None
        if (self.mdp.isTerminal(state) or (not len(actions_from_state) > 0)):
            max_q_value = 0
        else:
            for action in actions_from_state:
                q_value = self.computeQValueFromValues(state, action)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Do note that runValueIteration for all value iteration agents and computeActionFromValues could be refactored to all use a single method
        for idx in range(self.iterations):
            state_list = self.mdp.getStates()
            state_idx = idx % len(state_list)
            state = state_list[state_idx]
            actions_from_state = self.mdp.getPossibleActions(state)
            max_q_value = -float('inf')
            if (self.mdp.isTerminal(state) or (not len(actions_from_state) > 0)):
                max_q_value = 0
            else:
                for action in actions_from_state:
                    q_value = self.computeQValueFromValues(state, action)
                    if q_value > max_q_value:
                        max_q_value = q_value
            self.values[state] = max_q_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        def compute_diff(state):
            q_values = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
            diff = abs(self.getValue(state) - max(q_values))
            return diff

        state_list = self.mdp.getStates()

        # Compute predecessors of all states
        predecessor_dict = {}
        for state in state_list:
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if not prob > 0:
                        continue
                    if next_state not in predecessor_dict:
                        predecessor_dict[next_state] = set()
                    predecessor_dict[next_state].add(state)
        
        print('predecessor_dict is {0}'.format(predecessor_dict))

        # Initialize priority queue 
        priority_queue = util.PriorityQueue()

        for state in state_list:
            if self.mdp.isTerminal(state):
                continue
            diff = compute_diff(state) 
            priority_queue.update(state, -diff)

        for idx in range(self.iterations):
            if priority_queue.isEmpty():
                return
            state = priority_queue.pop()
            if not self.mdp.isTerminal(state):
                # Update value of this state if not terminal state
                actions_from_state = self.mdp.getPossibleActions(state)
                max_q_value = -float('inf')
                if not len(actions_from_state) > 0:
                    max_q_value = 0
                else:
                    for action in actions_from_state:
                        q_value = self.computeQValueFromValues(state, action)
                        if q_value > max_q_value:
                            max_q_value = q_value
                self.values[state] = max_q_value
            for predecessor in predecessor_dict[state]:
                diff = compute_diff(predecessor)
                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)
            

