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
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave))),

    # Rule: either one, but not both nor neither
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),

    # And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    # And(Or(CKnight, CKnave), Not(And(CKnight, CKnave))),
    # Rule: same rule, applied differently
    # And(Or(AKnight, AKnave), Implication(AKnight, Not(AKnave))),
    # And(Or(BKnight, BKnave), Implication(BKnight, Not(BKnave))),
    # And(Or(CKnight, CKnave), Implication(CKnight, Not(CKnave)))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave))),

    # Rule: either one, but not both nor neither
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave)))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    Implication(AKnight, Or(And(AKnave, BKnave), And(AKnight, BKnight))),
    Implication(AKnave, Not(Or(And(AKnave, BKnave), And(AKnight, BKnight)))),

    Implication(BKnight, Or(And(AKnight, BKnave), And(BKnight, AKnave))),
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(BKnight, AKnave)))),

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

    # Or(Implication(AKnight, AKnight),
    # Implication(AKnave, Not(AKnight)),
    # Implication(AKnight, AKnave),
    # Implication(AKnave, Not(AKnave))),

    # Implication(BKnight, And(Or(Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave))), Not(And(Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave)))))),
    # Implication(BKnave, Not(And(Or(Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave))), Not(And(Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave))))))),

    Implication(BKnight, (Or(Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave))))),
    Implication(BKnave, Not(Or(Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave))))),

    Implication(BKnight, CKnave),
    Implication(BKnave, Not(CKnave)),

    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight)),

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
