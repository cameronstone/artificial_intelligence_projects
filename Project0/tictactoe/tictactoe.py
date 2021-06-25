"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    # Game is over
    if terminal(board):
        return None

    # Count number of occurences of X and O
    x_count = 0
    o_count = 0
    for row in board:
        for box in row:
            if box == X:
                x_count = x_count + 1
            elif box == O:
                o_count = o_count + 1
    # When move count is tied, X is next
    if x_count <= o_count:
        return X
    # When X has moved once more than O, next move is O
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    possible_actions = set()
    # Track row index
    i = -1
    # Iterate through each row in the board
    for row in board:
        i = i + 1
        # Track row item index
        j = -1
        # Iterate through each item in a row
        for box in row:
            j = j + 1
            # Add every item (not yet visited) as a possibility
            if box is None:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # Create completely new board
    temp_board = copy.deepcopy(board)
    # Location of move to be made
    row_index = action[0]
    col_index = action[1]

    # Check for valid action
    if not 0 <= row_index <= 2 or not 0 <= col_index <= 2:
        raise Exception("Invalid Action")

    # Make move and update board
    if board[row_index][col_index] is None:
        temp_board[row_index][col_index] = player(board)
    else:
        raise Exception("Invalid Action")

    return temp_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    # Check for horizontal wins
    for row in board:
        if row[0] == row[1] == row[2] and row[0] is not None:
            return row[0]

    # Check for vertical wins
    for i in range(3):
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None:
            return board[0][i]

    # Check for diagonal wins
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]

    # If there is no winner, return None
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    # No winner case
    if winner(board) is None:
        for row in board:
            for box in row:
                # The game is not over
                if box is None:
                    return False
        # Scratch game
        return True
    else:
        # There was a winner
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    # Determine winner
    victor = winner(board)

    # Assign proper values accordingly
    if victor == X:
        return 1
    elif victor == O:
        return -1
    else:
        return 0


# Simulates O's optimal decision making
def min_value(board):
    v = math.inf
    if terminal(board):
        return utility(board)
    # Find the minimum of the values that the maximizing player would choose
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v


# Simulates X's optimal decision making
def max_value(board):
    v = -(math.inf)
    if terminal(board):
        return utility(board)
    # Find the maximum of the values that the minimizing player would choose
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    # If terminal board, return None
    if terminal(board):
        return None

    # Determine whose turn it is and the possible actions they can make
    current_player = player(board)
    moves = actions(board)

    if current_player == "X":
        utility_from_action = list()
        # Create pairings between the possible move and its eventual outcome
        for action in moves:
            utility_from_action.append((min_value(result(board, action)), action))
        temp = -(math.inf)
        optimal_move = None
        # Find the action that leads to maximized utility
        for (utility, action) in utility_from_action:
            if utility > temp:
                temp = utility
                optimal_move = action
        return optimal_move

    elif current_player == "O":
        utility_from_action = list()
        # Create pairings between the possible move and its eventual outcome
        for action in moves:
            utility_from_action.append((max_value(result(board, action)), action))
        temp = math.inf
        optimal_move = None
        # Find the action that leads to minimized utility
        for (utility, action) in utility_from_action:
            if utility < temp:
                temp = utility
                optimal_move = action
        return optimal_move

    else:
        return None
