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
        for _ in range(self.iterations):
            new_values = util.Counter()

            for state in self.mdp.getStates():
                max_q_value = float('-inf')

                for action in self.mdp.getPossibleActions(state):
                    q_value = sum(
                        prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
                    )

                    max_q_value = max(max_q_value, q_value)

                new_values[state] = max_q_value if max_q_value != float('-inf') else 0.0

            self.values = new_values

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
        q_value = sum(
            prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
            for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)
        )
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
        if self.mdp.isTerminal(state):
            return None

        max_action = max(self.mdp.getPossibleActions(state), key=lambda a: self.computeQValueFromValues(state, a))
        return max_action

        util.raiseNotDefined()

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
        for item in range(self.iterations):

            state = self.mdp.getStates()[item % len(self.mdp.getStates())]
            best = self.computeActionFromValues(state)
            if best is None:
                V = 0
            else:
                V = self.computeQValueFromValues(state, best)
            self.values[state] = V

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
        all_states = self.mdp.getStates()
        predecessors = {state: set() for state in all_states}

        for state in all_states:
            for action in self.mdp.getPossibleActions(state):
                possible_next_states = self.mdp.getTransitionStatesAndProbs(state, action)

                for next_state, pred_prob in possible_next_states:
                    if pred_prob > 0:
                        predecessors[next_state].add(state)

        priority_queue = util.PriorityQueue()

        for state in all_states:
            if self.mdp.isTerminal(state):
                continue

            max_q_value = max(self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state))
            diff = abs(self.values[state] - max_q_value)
            priority_queue.push(state, -diff)

        for _ in range(self.iterations):
            if priority_queue.isEmpty():
                break

            state = priority_queue.pop()

            if not self.mdp.isTerminal(state):
                self.values[state] = max(
                    self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state))

            for predecessor in predecessors[state]:
                max_q_value = max(
                    self.computeQValueFromValues(predecessor, a) for a in self.mdp.getPossibleActions(predecessor))
                diff = abs(self.values[predecessor] - max_q_value)

                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)

