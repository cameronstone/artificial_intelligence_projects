import itertools
import random
import copy


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
        if len(self.cells) == self.count:
            mines.append(self.cells)
        return mines

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        safe = set()
        if self.count == 0:
            safe.append(self.cells)
        return safe

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
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
        self.moves_made.add(cell)
        for sent in self.knowledge:
            sent.mark_safe(cell)
        self.safes.add(cell)

        def generate_neighbors(cell):
            num_mines_removed = 0
            neighbors = set()
            for (i, j) in cell:
                for row in range(-1, 2):
                    for col in range(-1, 2):
                        new_i = i + row
                        new_j = j + col
                        if -1 < new_i < self.height and -1 < new_i < self.height:
                            neighbors.add(new_i, new_j)
            for unit in self.moves_made:
                if unit in neighbors:
                    neighbors.remove(unit)
            for unit in self.safe:
                if unit in neighbors:
                    neighbors.remove(unit)
            for unit in self.mines:
                if unit in neighbors:
                    neighbors.remove(unit)
                    num_mines_removed = num_mines_removed + 1
            return (neighbors, num_mines_removed)

        (my_neighbors, count_reduction) = generate_neighbors(cell)
        sentence = Sentence(cells=my_neighbors, count=count - count_reduction)
        self.knowledge.append(sentence)

        sets_of_safes = set()
        for sent in self.knowledge:
            sets_of_safes.add(sent.known_safes())
        
        # knowledge_base = copy.deepcopy(self.knowledge)
        for sent in self.knowledge:
            for safes in sets_of_safes:
                for safe in safes:
                    sent.mark_safe(safe)

        sets_of_mines = set()
        for sent in self.knowledge:
            sets_of_mines.add(sent.known_mines())

        for sent in self.knowledge:
            for mines in sets_of_mines:
                for mine in mines:
                    sent.mark_mine(mine)

        # deepcopy?
        # add or just edit the existing set
        for sent1 in self.knowledge:
            for sent2 in self.knowledge:
                if sent1 != sent2:
                    if sent1.issubset(sent2):
                        new_sent_cells = sent2.cells - sent1.cells
                        new_sent_count = sent2.count - sent1.count
                        self.knowledge.append(Sentence(cells=new_sent_cells, count=new_sent_count))


    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        raise NotImplementedError

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        raise NotImplementedError
