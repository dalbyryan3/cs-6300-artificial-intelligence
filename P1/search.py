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

from xml.sax.handler import feature_string_interning
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

def generalSearch(problem, initial_fringe, push_with_priority=False):
    closed_states = set()
    fringe = initial_fringe
    while(True):
        if fringe.isEmpty():
            # print("Fringe empty, no goal found")
            return False
        node = fringe.pop()
        node_state, actions_to_node, cost_to_node = node
        if (problem.isGoalState(node_state)):
            # print("Found goal state")
            # print("Goal node state value: {0}".format(node_state))
            # print("Actions to node: {0}\nLength: {1}".format(actions_to_node, len(actions_to_node)))
            # print("Goal node total cost: {0}".format(cost_to_node))
            return actions_to_node
        if (node_state not in closed_states):
            closed_states.add(node_state)
            for successor in problem.getSuccessors(node_state):
                successor_node_state, successor_action, successor_cost = successor

                successor_actions_to_node = actions_to_node.copy()
                successor_actions_to_node.append(successor_action)

                successor_cost_to_node = cost_to_node
                successor_cost_to_node += successor_cost

                if push_with_priority:
                    fringe.push((successor_node_state, successor_actions_to_node, successor_cost_to_node), successor_cost_to_node)
                else:
                    fringe.push((successor_node_state, successor_actions_to_node, successor_cost_to_node))

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
    initial_fringe = util.Stack()
    initial_fringe.push((problem.getStartState(), [], 0))
    result = generalSearch(problem, initial_fringe)
    return result

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    initial_fringe = util.Queue()
    initial_fringe.push((problem.getStartState(), [], 0))
    result = generalSearch(problem, initial_fringe)
    return result


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    initial_fringe = util.PriorityQueue()
    initial_fringe.push((problem.getStartState(), [], 0), 0)
    result = generalSearch(problem, initial_fringe, push_with_priority=True)
    return result

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    # Use a PriorityQueueWithFunction, may have to update generalSearch
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
