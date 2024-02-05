# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    # Initialize the stack with the starting state and an empty path
    stack = Stack()
    stack.push((problem.getStartState(), []))

    # Initialize a set to keep track of visited states
    visited = set()

    while not stack.isEmpty():
        state, path = stack.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    new_path = path + [action]
                    stack.push((successor, new_path))

    # If no solution is found, return an empty list
    return []

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    # Initialize the queue with the starting state and an empty path
    queue = Queue()
    queue.push((problem.getStartState(), []))

    # Initialize a set to keep track of visited states
    visited = set()

    while not queue.isEmpty():
        state, path = queue.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    new_path = path + [action]
                    queue.push((successor, new_path))

    # If no solution is found, return an empty list
    return []

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    # Initialize the priority queue with the starting state and an empty path
    queue = PriorityQueue()
    queue.push((problem.getStartState(), []), 0)

    # Initialize a set to keep track of visited states
    visited = set()

    while not queue.isEmpty():
        state, path = queue.pop()

        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.add(state)

            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    new_path = path + [action]
                    cost = problem.getCostOfActions(new_path)
                    queue.push((successor, new_path), cost)

                # State is in queue. Check if current path is cheaper from the previous one
                elif successor not in visited:
                    for item in queue.heap:
                        if item[2][0] == successor:
                            old_cost = problem.getCostOfActions(item[2][1])

                    new_cost = problem.getCostOfActions(path + [action])

                    # State is cheaper with its new parent -> update and fix parent
                    if old_cost > new_cost:
                        new_path = path + [action]
                        queue.update((successor, new_path), new_cost)

    # If no solution is found, return an empty list
    return []

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    # Create a priority queue using PriorityQueue
    queue = PriorityQueue()

    # Initialize visited dictionary to keep track of visited states
    visited = {}

    # Check if the initial state is the goal state
    if problem.isGoalState(problem.getStartState()):
        return []

    # Initialize the queue with the starting state and an empty path
    start_state = problem.getStartState()
    queue.push((start_state, [], 0), 0)

    while not queue.isEmpty():
        state, path, cost = queue.pop()

        # Terminate condition: reach the goal state
        if problem.isGoalState(state):
            return path

        # Skip this state if it has already been visited with a lower cost
        if state in visited and visited[state] <= cost:
            continue

        visited[state] = cost

        # Get successors of the current state
        successors = problem.getSuccessors(state)

        if successors:
            for next_state, action, step_cost in successors:
                if next_state not in visited:
                    new_path = path + [action]
                    new_cost = cost + step_cost
                    priority = new_cost + heuristic(next_state, problem)
                    queue.push((next_state, new_path, new_cost), priority)

    # Terminate condition: can't find a solution
    return []

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
