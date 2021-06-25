from tictactoe import player
from tictactoe import initial_state
from tictactoe import actions
from tictactoe import result
from tictactoe import winner
from tictactoe import terminal
from tictactoe import utility

EMPTY = None

empty_board = initial_state()
x_turn = [["O", "X", EMPTY],
            [EMPTY, "O", EMPTY],
            [EMPTY, EMPTY, "X"]]

o_turn = [[EMPTY, EMPTY, EMPTY],
            ["X", EMPTY, EMPTY],
            ["O", "X", EMPTY]]

filled_board = [["O", EMPTY, "X"],
            ["X", "O", "O"],
            ["O", "X", "X"]]


filled_set = set()
filled_set.add((0,1))

print(player(empty_board) == "X")
print(player(x_turn) == "X")
print(player(o_turn) == "O")


o_set = set()
o_set.add((0,1))
o_set.add((0,0))
o_set.add((0,2))
o_set.add((1,1))
o_set.add((1,2))
o_set.add((2,2))

print(actions(filled_board) == filled_set)
print(actions(o_turn) == o_set)


o_turn_next = [["O", EMPTY, EMPTY],
            ["X", EMPTY, EMPTY],
            ["O", "X", EMPTY]]

x_turn_next = [["O", "X", EMPTY],
            [EMPTY, "O", EMPTY],
            [EMPTY, "X", "X"]]

print(result(o_turn, (0,0)) ==  o_turn_next)
print(result(x_turn, (2,1)) ==  x_turn_next)
# print(result(x_turn, (0, 0)) ==  x_turn_next)
# print(result(x_turn, (2, 3)) ==  x_turn_next)

x_vert = [["O", "X", EMPTY],
            [EMPTY, "X", "O"],
            ["O", "X", "X"]]

o_hor = [["O", "X", "X"],
            ["O", "O", "O"],
            [EMPTY, "X", "X"]]

o_diag = [["O", "X", EMPTY],
            [EMPTY, "O", EMPTY],
            ["X", "X", "O"]]

none_winner = [["O", "X", EMPTY],
                [EMPTY, "O", EMPTY],
                [EMPTY, "X", "X"]]
                

print(winner(x_vert) == "X")
print(winner(o_hor) == "O")
print(winner(o_diag) == "O")
print(winner(none_winner) is None)


scratch = [["X", "X", "O"],
            ["O", "O", "X"],
            ["X", "O", "X"]]

print(terminal(x_vert))
print(terminal(o_hor))
print(terminal(o_diag))
print(not terminal(none_winner))
print(terminal(scratch))

print(utility(x_vert) == 1)
print(utility(o_hor) == -1)
print(utility(o_diag) == -1)
print(utility(scratch) == 0)