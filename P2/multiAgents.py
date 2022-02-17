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
        # print("successorGameState:") # This is the full game state 
        # print(successorGameState) 
        # print("newPos:") # This is the new position of pacman
        # print(newPos)
        # print("newFood:") # This is a boolean grid of the map where food is
        # print(newFood)
        # print(newFood.asList())
        # print("newGhostStates:") # Like pacman state but for all ghosts stored in a collection
        # print(newGhostStates[0])
        # print("newScaredTimes:") # This is the time left for the ghosts to be scared run
        # print(newScaredTimes)

        def get_avg_food_distance_sum(food):
            food_list = food.asList()
            food_sum = 0
            count = 0
            for f in food_list:
                food_sum += manhattanDistance(newPos, f)
                count += 1
            return food_sum/count if count > 0 else 0

        def get_avg_ghost_distance_sum(ghosts):
            ghost_sum = 0
            count = 0
            for g in ghosts:
                ghost_sum += manhattanDistance(newPos, g.getPosition())
                count += 1
            return ghost_sum/count if count > 0 else 0

        def are_all_scared(scared_times):
            for s in scared_times:
                if s == 0:
                    return False
            return True
                
        # print("start")
        # print(get_avg_food_distance_sum(newFood))
        # print(get_avg_ghost_distance_sum(newGhostStates))
        # print(successorGameState.getScore())
        # print("end")

        game_score_w = 4
        food_w = 4 
        ghost_w = 1 
        if are_all_scared(newScaredTimes):
            game_score_w *= 10
            food_w *= 10
            ghost_w *= 0
        evaluation = (game_score_w*successorGameState.getScore()) - (food_w*get_avg_food_distance_sum(newFood)) + (ghost_w*get_avg_ghost_distance_sum(newGhostStates))
        return evaluation

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
        Returns the total number of agents in the gam

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def get_next_agent_index_and_depth(state, agent_index, current_depth):
            if agent_index == state.getNumAgents()-1: # We just expanded final agent 
                next_agent_index = 0
                next_depth = current_depth + 1
            else:
                next_agent_index = agent_index + 1
                next_depth = current_depth
            return next_agent_index, next_depth

        def recursive_minimax(state, agent_index, depth):
            if depth == self.depth: # At this point have completed current_depth number of turns (search plys) so should stop if we are at the set depth
                return self.evaluationFunction(state), ""
            legal_actions = state.getLegalActions(agent_index)

            if len(legal_actions) == 0: # Terminal state
                return self.evaluationFunction(state), ""

            is_max = (agent_index == 0) # If our current level is a max node, otherwise we are at a min node
            best_value = -float('inf') if is_max else float('inf')
            best_action = ""
            next_agent_index, next_depth = get_next_agent_index_and_depth(state, agent_index, depth)
            for action in legal_actions:
                next_game_state = state.generateSuccessor(agent_index, action)
                value, _ = recursive_minimax(next_game_state, next_agent_index, next_depth)
                if ((is_max) and (value > best_value)) or ((not is_max) and (value < best_value)):
                    best_value = value 
                    best_action = action
            return best_value, best_action
        
        minimax_value, minimax_action = recursive_minimax(gameState, 0, 0)

        return minimax_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
