from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    
    # Consider what A said and when that is true
    Biconditional(AKnight, And(AKnight, AKnave)),

    # Rule: either one, but not both nor neither
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),

)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    
    # Consider what A said and when that is true
    Biconditional(AKnight, And(AKnave, BKnave)),

    # Rule: either one, but not both nor neither
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave)))

)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    
    # Consider what A said and when that is true
    Biconditional(AKnight, Or(And(AKnave, BKnave), And(AKnight, BKnight))),

    # Consider what B said and when that is true
    Biconditional(BKnight, Or(And(AKnight, BKnave), And(BKnight, AKnave))),

    # Rule: either one, but not both nor neither
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave)))

)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(

    # Nested Biconditionals (what A said depends on if what B said is true or not)
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),

    # Consider what else B said
    Biconditional(BKnight, CKnave),

    # Consider what C said
    Biconditional(CKnight, AKnight),

    # Rule: either one, but not both nor neither
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    And(Or(CKnight, CKnave), Not(And(CKnight, CKnave)))
    
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
