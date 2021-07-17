import sys
import random
import copy

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # create a copy to apply changes to
        domains_copy = copy.deepcopy(self.domains)
        # loop through all variables in domains dict
        for var in self.domains.keys():
            # loop through all words in that dict key's values
            for word in self.domains[var]:
                # if word doesn't match the variable's length, remove
                if var.length != len(word):
                    domains_copy[var].remove(word)
        self.domains = domains_copy

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        # create indicator for revisions made
        revision = False
        # check that they have an overlap
        overlap = self.crossword.overlaps[x, y]
        if overlap is not None:
            # get indices of each variable's character that overlaps
            (index_x, index_y) = overlap
            # make a copy of x_domains to apply changes
            x_domains_copy = copy.deepcopy(self.domains[x])
            # loop through every possible value of x
            for word_x in self.domains[x]:
                word_available = False
                # loop through every possible value of y
                for word_y in self.domains[y]:
                    # if y has a viable option, indicate it
                    if word_x[index_x] == word_y[index_y]:
                        word_available = True
                # if no words in y are viable, remove from x's domain
                if not word_available:
                    x_domains_copy.remove(word_x)
                    revision = True
            self.domains[x] = x_domains_copy
        return revision

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # if arcs is None, add all possible arcs
        if arcs is None:
            queue = []
            for v1 in self.crossword.variables:
                for v2 in self.crossword.variables:
                    if v1 != v2:
                        queue.append((v1, v2))
        # otherwise, use arcs as initial queue
        else:
            queue = arcs
        # loop until the queue is empty
        while queue != []:
            # make one arc consistent at a time
            (x, y) = queue.pop(0)
            if self.revise(x, y):
                # if a variable's domain is reduced to 0, no solution
                if len(self.domains[x]) == 0:
                    return False
                # is a revision was made, add all neighbors but y to queue
                else:
                    for neighbor in self.crossword.neighbors(x):
                        if neighbor != y:
                            queue.append((neighbor, x))
        # everything is arc consistent
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # check that assignment dictionary has all variables
        if len(assignment.keys()) == len(self.crossword.variables):
            # check that there is a value assigned
            for var in assignment.keys():
                if assignment[var] is None:
                    return False
            return True
        # not complete
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        assigned_words = assignment.values()

        # check that all values are distinct
        for var1 in assigned_words:
            count = 0
            for var2 in assigned_words:
                if var1 == var2:
                    count += 1
            if count != 1:
                return False
        
        assigned_variables = assignment.keys()
        # check the every value is correct length
        for var in assigned_variables:
            if var.length != len(assignment[var]):
                return False

        # check that no conflicts between neighbors exist
        # loop through every assigned variable
        for var in assigned_variables:
            # loop through every neighbor of that variable
            for neighbor in self.crossword.neighbors(var):
                # check if that neighbor is assigned
                if neighbor in assigned_variables:
                    # get overlap
                    overlap = self.crossword.overlaps[var, neighbor]
                    if overlap is not None:
                        # get indices of each variable's character that overlaps
                        (index_var, index_neighbor) = overlap
                        # check if assigned words for those variables have a conflict
                        if assignment[var][index_var] != assignment[neighbor][index_neighbor]:
                            return False
        # passed all three constraints
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # this dictionary will have word keys and lcv values
        lcv_values = {}
        unassigned_neighbors = []

        # creates a list of unassigned neighbors for var
        for neighbor in self.crossword.neighbors(var):
            if neighbor not in assignment.keys():
                unassigned_neighbors.append(neighbor)

        # loop through every word in var's domain
        for word in self.domains[var]:
            lcv = 0
            # count every neighbor whose word would get ruled out
            for neighbor in unassigned_neighbors:
                if word in self.domains[neighbor]:
                    lcv += 1
            # add that lcv value paired with the key as the word
            lcv_values[word] = lcv

        # sort dictionary by ascending lcv value
        sorted_lcv = dict(sorted(lcv_values.items(), key= lambda word: word[1]))
        return sorted_lcv.keys()

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # find available, unassigned variables
        available_variables = self.crossword.variables.difference(assignment.keys())
        # set min_length tracker to an initial value
        first_var = available_variables.pop()
        min_length = len(self.domains[first_var])
        min_var = [first_var]
        # loop through all available variables
        for var in available_variables:
            # if a tie, append
            if len(self.domains[var]) == min_length:
                min_var.append(var)
            # if new minimum, reset list to single variable
            elif len(self.domains[var]) < min_length:
                min_var = [var]
        # if there is a variable with minimum remaining values, return it
        if len(min_var) == 1:
            return min_var[0]
        # if there is a tie for minimum remaining values, check degree 
        else:
            # create counter for current max number of neighbors
            max_neighbors = 0
            most_neighbors = []
            # loop through each variable in the tie
            for var in min_var:
                num_neighbors = len(self.crossword.neighbors(var))
                # if its number of neighbors matches current max, append
                if num_neighbors == max_neighbors:
                    most_neighbors.append(var)
                # if it is outright newest max, reset to single variable
                elif num_neighbors > max_neighbors:
                    most_neighbors = [var]
            # if there is a variable with the highest degree, return it
            if len(most_neighbors) == 1:
                return most_neighbors[0]
            # there is a tie in variables with highest degree, return random
            else:
                return random.choice(most_neighbors)

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # if the assignment is complete and consistent, return it
        if self.assignment_complete(assignment):
            if self.consistent(assignment):
                return assignment
        # select an unassigned variable (based on heuristics)
        var = self.select_unassigned_variable(assignment)
        # loop through every word in ascending order (based on heuristic)
        for word in self.order_domain_values(var, assignment):

                # add assignment to COPY
                new_assignment = copy.deepcopy(assignment)
                new_assignment[var] = word
                # allow for rewind of ac3's affects
                domain = copy.deepcopy(self.domains)

                # ensure the word is not already used
                if self.consistent(new_assignment):

                    # creates inferences
                    neighbors = self.crossword.neighbors(var)
                    arcs = []
                    for neighbor in neighbors:
                        arcs.append((neighbor, var))
                    inferences = self.ac3(arcs)

                    # adds inferences to assignment (if consistent)
                    if inferences:

                        # recursively call backtrack to see if we find solution
                        result = self.backtrack(new_assignment)

                        # if result is not a failure, return it
                        if result is not None:
                            return result

                # if it doesn't yield a solution, backtrack by removing assignment
                new_assignment.popitem()
                # removes inferences from assignment
                self.domains = domain

        # if we run out of variables and words to try, return None
        return None

def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
