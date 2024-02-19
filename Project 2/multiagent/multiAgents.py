# multiAgents.py
# --------------
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

from functools import partial
from util import manhattanDistance
from game import Directions
import random, util


from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodpos = newFood.asList()
        foodAgentDist = [manhattanDistance(newPos, fpos) for fpos in foodpos]
        if len(foodAgentDist) == 0:
            minFoodDist = 0.001
        else:
            minFoodDist = min(foodAgentDist)

        ghostPos = [gs.configuration.pos for gs in newGhostStates]
        ghostAgentDist = [manhattanDistance(newPos, ghostPos[i]) for i in range(len(ghostPos)) if newScaredTimes[i]==0]
        if len(ghostAgentDist) == 0:
            minGhostDist = newFood.width*newFood.height
        else:
            minGhostDist = max(min(ghostAgentDist), 2)

        return successorGameState.getScore() + 10.0/minFoodDist - 20.0/minGhostDist

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(gs, agentId, depth):
            # print("value, agent: %d, depth: %d"%(agentId, depth))
            if gs.isWin() or gs.isLose() or depth==0:
                # print("Terminal State")
                return (self.evaluationFunction(gs), None)
            
            if agentId == 0:
                return max_value(gs, agentId, depth)
            elif agentId <= gs.getNumAgents() - 1:
                return min_value(gs, agentId, depth)

        def max_value(gs, agentId, depth):
            # print("max-value, agent: %d, depth: %d"%(agentId, depth))
            v = -float("inf")
            max_a = None

            for action in gs.getLegalActions(agentId):
                new_gs = gs.generateSuccessor(agentId, action)
                new_v,_ = value(new_gs, 1, depth)
                if new_v > v:
                    v = new_v
                    max_a = action
            return v, max_a
        
        def min_value(gs, agentId, depth):
            # print("min-value, agent: %d, depth: %d"%(agentId, depth))
            v = float("inf")
            min_a = None
            lastGhostId = gs.getNumAgents() - 1

            for action in gs.getLegalActions(agentId):
                new_gs = gs.generateSuccessor(agentId, action)

                if agentId < lastGhostId:
                    new_v,_ = value(new_gs, agentId+1, depth)
                elif agentId == lastGhostId:
                    new_v,_ = value(new_gs, 0, depth-1)
                else:
                    new_v = float("inf") # This should not happen
                
                if new_v < v:
                    v = new_v
                    min_a = action
            return v, min_a
        
        _, max_action = value(gameState, 0, self.depth)
        return max_action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(gs, agentId, depth, alpha, beta):
            # print("value, agent: %d, depth: %d"%(agentId, depth))
            if gs.isWin() or gs.isLose() or depth==0:
                # print("Terminal State")
                return (self.evaluationFunction(gs), None)
            
            if agentId == 0:
                return max_value(gs, agentId, depth, alpha, beta)
            elif agentId <= gs.getNumAgents() - 1:
                return min_value(gs, agentId, depth, alpha, beta)

        def max_value(gs, agentId, depth, alpha, beta):
            # print("max-value, agent: %d, depth: %d"%(agentId, depth))
            v = -float("inf")
            max_a = None

            for action in gs.getLegalActions(agentId):
                new_gs = gs.generateSuccessor(agentId, action)
                new_v,_ = value(new_gs, 1, depth, alpha, beta)
                if new_v > v:
                    v = new_v
                    max_a = action
                if v > beta:
                    return v, max_a
                alpha = max(alpha, v)
            return v, max_a
        
        def min_value(gs, agentId, depth, alpha, beta):
            # print("min-value, agent: %d, depth: %d"%(agentId, depth))
            v = float("inf")
            min_a = None
            lastGhostId = gs.getNumAgents() - 1

            for action in gs.getLegalActions(agentId):
                new_gs = gs.generateSuccessor(agentId, action)

                if agentId < lastGhostId:
                    new_v,_ = value(new_gs, agentId+1, depth, alpha, beta)
                elif agentId == lastGhostId:
                    new_v,_ = value(new_gs, 0, depth-1, alpha, beta)
                else:
                    new_v = float("inf") # This should not happen
                
                if new_v < v:
                    v = new_v
                    min_a = action
                if v < alpha:
                    return v, min_a
                beta = min(beta, v)
            return v, min_a
        
        _, max_action = value(gameState, 0, self.depth, -float("inf"), float("inf"))
        return max_action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        max_value = float('-inf')
        max_action = Directions.STOP

        for action in gameState.getLegalActions(0):
            successor_state = gameState.generateSuccessor(0, action)
            successor_value = self.get_node_value(successor_state, 0, 1)
            if successor_value > max_value:
                max_value, max_action = successor_value, action

        return max_action

    def get_node_value(self, game_state, current_depth, agent_index):
        """
        Calculate the value of a node in the game tree using alpha-beta pruning.
        """
        if current_depth == self.depth or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        if agent_index == 0:
            return self.max_value(game_state, current_depth)
        else:
            return self.exp_value(game_state, current_depth, agent_index)

    def max_value(self, game_state, current_depth):
        """
        Calculate the maximum value for a max agent (Pacman) node.
        """
        max_value = float('-inf')
        for action in game_state.getLegalActions(0):
            successor_state = game_state.generateSuccessor(0, action)
            successor_value = self.get_node_value(successor_state, current_depth, 1)
            max_value = max(max_value, successor_value)
        return max_value

    def exp_value(self, game_state, current_depth, agent_index):
        """
        Calculate the expected value for a min agent (Ghost) node.
        """
        total_value = 0.0
        num_agents = game_state.getNumAgents()
        for action in game_state.getLegalActions(agent_index):
            successor_state = game_state.generateSuccessor(agent_index, action)
            if agent_index == num_agents - 1:
                successor_value = self.get_node_value(successor_state, current_depth + 1, 0)
            else:
                successor_value = self.get_node_value(successor_state, current_depth, agent_index + 1)
            total_value += successor_value

        num_actions = len(game_state.getLegalActions(agent_index))
        return total_value / num_actions
       

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Get the current position of Pacman,food layout, and the ghost states
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Constants for evaluation
    INF = 100000000.0  # Infinite value
    WEIGHT_FOOD = 10.0  # Food base value
    WEIGHT_GHOST = -10.0  # Ghost base value
    WEIGHT_SCARED_GHOST = 100.0  # Scared ghost base value

    # Initialize the score based on the current game score
    score = currentGameState.getScore()

    # Evaluate the distance to the closest food
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]

    if len(distancesToFoodList) > 0:
        # Add a bonus based on the closest food distance
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        # If there's no food left, add a fixed bonus
        score += WEIGHT_FOOD

    # Evaluate the distance to ghosts
    for ghost in newGhostStates:
        distance = util.manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:
                # If the ghost is scared, add a bonus based on distance
                score += WEIGHT_SCARED_GHOST / distance
            else:
                # If the ghost is not scared, penalize based on distance
                score += WEIGHT_GHOST / distance
        else:
            # If Pacman is in the same position as a ghost, return a large negative value
            return -INF

    return score

  

# Abbreviation
better = betterEvaluationFunction
