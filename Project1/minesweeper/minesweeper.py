import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        mines = set()
        # When the cells equal the they are all mines
        if len(self.cells) == self.count:
            mines.update(self.cells)
        return mines

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        safe = set()
        # When count is 0, we know they are safe
        if self.count == 0:
            safe.update(self.cells)
        return safe

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            # When removing cells, also adjust the count
            self.cells.remove(cell)
            self.count = self.count - 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # Mark cell as a move made
        self.moves_made.add(cell)
        # Mark cell as safe
        self.mark_safe(cell)

        # Generates all undetermined neighbors
        def generate_neighbors(cell):
            num_mines_removed = 0
            neighbors = set()
            # Generate all valid neighbors
            for row in range(-1, 2):
                for col in range(-1, 2):
                    i = cell[0] + row
                    j = cell[1] + col
                    if 0 <= i <= self.height - 1 and 0 <= j <= self.width - 1:
                        neighbors.add((i, j))
            # Removes safes
            for unit in self.safes:
                if unit in neighbors:
                    neighbors.remove(unit)
            # Remove mines
            for unit in self.mines:
                if unit in neighbors:
                    neighbors.remove(unit)
                    # When removing mines, pass a number to adjust the count
                    num_mines_removed = num_mines_removed + 1
            return (neighbors, num_mines_removed)

        # Create new sentence based off of undetermined neighbors
        (my_neighbors, count_reduction) = generate_neighbors(cell)
        sentence = Sentence(cells=my_neighbors, count=count - count_reduction)
        self.knowledge.append(sentence)

        # Create boolean to indicate when something changed (repeat loop)
        something_changed = True
        while something_changed:
            something_changed = False

            # Update set of known safe cells
            for sent in self.knowledge:
                known = sent.known_safes()
                if known != set():
                    for cell in known:
                        self.mark_safe(cell)

            # Update set of known mines
            for sent in self.knowledge:
                known = sent.known_mines()
                if known != set():
                    for cell in known:
                        self.mark_mine(cell)

            # Always filter out empty sets
            filt = filter(lambda s: s.cells != set(), self.knowledge)
            self.knowledge = list(filt)

            # Iterate through all combinations of sentences from KB
            combos = list(itertools.permutations(self.knowledge, 2))
            for (sent1, sent2) in combos:
                # When true subset, make new inference
                if sent1.cells.issubset(sent2.cells) and sent1 != sent2:
                    new_cells = sent2.cells - sent1.cells
                    new_count = sent2.count - sent1.count
                    inf = Sentence(cells=new_cells, count=new_count)
                    # Make sure knowledge being added is new
                    if inf not in self.knowledge or inf.cells == set():
                        self.knowledge.append(inf)
                        # Indicate change, try new resolves
                        something_changed = True

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        keep_trying = True
        # Try to find a safe move
        while keep_trying:
            if len(self.safes) == 0:
                keep_trying = False
                return None
            # Ensure the move hasn't already been made
            move = self.safes.pop()
            if move not in self.moves_made:
                keep_trying = False
                return move

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        available_cells = set()
        # Create set of all possible cells not already made
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.moves_made and (i, j) not in self.mines:
                    available_cells.add((i, j))
        if len(available_cells) == 0:
            return None
        else:
            # Pick random move
            move = random.sample(available_cells, 1)
            return move[0]
