"""G4 reasoning generators (E2) — pure stdlib, answers from seed only.

Families: S5 permutation composition, templated arithmetic (clause
tiers + NoOp distractor + held-out operand ranges), ProofWriter-style
deductive closure depth 0–3, boolean expressions, Dyck-k with a length
split, modular arithmetic (ID vs held-out OOD operand range), small-N
Knights & Knaves. All items are likelihood-scoreable MC — the
small-scale design rule from research/10 §1, §7.
"""

import itertools
import random

from .generators import _item, _numeric_choices, digits, lexicon

_NAMES = [
    "anna", "ben", "carla", "david", "emma", "felix", "gina", "hugo",
    "irene", "jack", "kate", "liam", "mona", "noah", "olga", "paul",
]
_OBJECTS = [
    "apples", "marbles", "stickers", "coins", "books", "shells", "cards",
]
_COLORS = ["red", "blue", "green", "yellow", "silver"]


# ------------------------------------------------------------ S5 word problem


def _rand_perm(rng):
    p = list(range(5))
    rng.shuffle(p)
    return p


def s5_compose(seed, k=5):
    """Compose k permutations of S5; query the product on one element.
    NC^1-complete probe (2402.12875 / 2404.08819): the signed blind spot
    of fixed-state architectures without a scratchpad."""
    rng = random.Random(int(seed))
    perms = [_rand_perm(rng) for _ in range(k)]
    x = rng.randrange(5)
    v = x
    for p in perms:
        v = p[v]
    lines = [f"p{i+1} = ({' '.join(str(t) for t in p)})" for i, p in enumerate(perms)]
    order = " then ".join(f"p{i+1}" for i in range(k))
    prompt = (
        "Each p maps 0 1 2 3 4 to the listed images.\n"
        + "\n".join(lines)
        + f"\nApply {order} to {x}. The result is"
    )
    choices = ["0", "1", "2", "3", "4"]
    return [
        _item("s5", prompt, choices, v, f"s5/k{k}", k=k, applied_to=x, answer=v)
    ]


# ------------------------------------------------------------ arithmetic


def arith(seed, tier=1, noop=False, extrap=False):
    """Templated word problems (GSM-Symbolic 2410.05229 knobs): clause
    tiers 1–3, an irrelevant NoOp clause variant, and an extrapolation
    split with operands held out of the dev family range."""
    rng = random.Random(int(seed))
    lo, hi = (61, 97) if extrap else (2, 25)
    a, b, c = (rng.randint(lo, hi) for _ in range(3))
    n1, n2 = rng.sample(_NAMES, 2)
    obj = rng.choice(_OBJECTS)
    clauses = [f"{n1} has {a} {obj}."]
    if tier == 1:
        clauses.append(f"{n2} gives {n1} {b} more {obj}.")
        ans = a + b
    elif tier == 2:
        clauses.append(f"{n2} gives {n1} {b} more {obj}.")
        clauses.append(f"later {n1} gives {c} {obj} to {n2}.")
        ans = a + b - c
    else:
        clauses.append(f"each of {n1}'s {b} friends gives {n1} {c} {obj}.")
        ans = a + b * c
    if noop:
        clauses.insert(1, f"{n1} likes the color {rng.choice(_COLORS)}.")
    clauses.append(f"how many {obj} does {n1} have now?")
    prompt = " ".join(clauses) + " answer:"
    choices, gold = _numeric_choices(rng, ans)
    tag = f"arith/t{tier}" + ("/noop" if noop else "") + ("/extrap" if extrap else "")
    return [_item("arith", prompt, choices, gold, tag, answer=ans)]


# ------------------------------------------------------------ ProofWriter-style


def proofwriter(seed, depth=1, positive=None):
    """Deductive closure over a generated fact + rule chain (2012.13048).
    Depth = rule hops needed; negatives query an attribute outside the
    closure (closed-world answer 'no')."""
    rng = random.Random(int(seed))
    ent = rng.choice(_NAMES)
    attrs = lexicon(rng, depth + 2)
    facts = [f"{ent} is {attrs[0]}."]
    rules = [
        f"if someone is {attrs[i]} then they are {attrs[i+1]}."
        for i in range(depth)
    ]
    pos = rng.random() < 0.5 if positive is None else positive
    target = attrs[depth] if pos else attrs[depth + 1]
    gold_word = "yes" if pos else "no"
    body = facts + rules
    rng.shuffle(body)
    prompt = " ".join(body) + f" question: is {ent} {target}? answer (yes or no):"
    return [
        _item(
            "proof",
            prompt,
            ["yes", "no"],
            0 if pos else 1,
            f"proof/d{depth}" + ("/pos" if pos else "/neg"),
        )
    ]


# ------------------------------------------------------------ boolean exprs


def _bool_tree(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        v = rng.random() < 0.5
        return ("lit", v)
    op = rng.choice(("and", "or", "not"))
    if op == "not":
        return ("not", _bool_tree(rng, depth - 1))
    return (op, _bool_tree(rng, depth - 1), _bool_tree(rng, depth - 1))


def _bool_eval(node):
    op = node[0]
    if op == "lit":
        return node[1]
    if op == "not":
        return not _bool_eval(node[1])
    if op == "and":
        return _bool_eval(node[1]) and _bool_eval(node[2])
    return _bool_eval(node[1]) or _bool_eval(node[2])


def _bool_render(node):
    op = node[0]
    if op == "lit":
        return "true" if node[1] else "false"
    if op == "not":
        return f"not ({_bool_render(node[1])})"
    return f"({_bool_render(node[1])}) {op} ({_bool_render(node[2])})"


def boolean_expr(seed, depth=2):
    rng = random.Random(int(seed))
    tree = _bool_tree(rng, depth)
    val = _bool_eval(tree)
    prompt = f"evaluate: {_bool_render(tree)} . answer (true or false):"
    return [
        _item("bool", prompt, ["true", "false"], 0 if val else 1, f"bool/d{depth}")
    ]


# ------------------------------------------------------------ Dyck-k


def dyck(seed, k=2, n_pairs=12, probes=3):
    """Dyck-k closing-bracket prediction with a length split (train-band
    ≤20 pairs / long 21–40 pairs, per 2210.10749 shortcut warning)."""
    rng = random.Random(int(seed))
    opens = ["(", "["][:k]
    closes = [")", "]"][:k]
    stack, seq = [], []
    while sum(1 for c in seq if c in opens) < n_pairs or stack:
        if stack and (rng.random() < 0.45 or sum(1 for c in seq if c in opens) >= n_pairs):
            seq.append(closes[opens.index(stack.pop())])
        else:
            b = rng.choice(opens)
            stack.append(b)
            seq.append(b)
    closing_pos = [i for i, c in enumerate(seq) if c in closes and i > 0]
    rng.shuffle(closing_pos)
    choices = closes + opens
    items = []
    for m in closing_pos[:probes]:
        gold_tok = seq[m]
        prompt = "type the next bracket: " + " ".join(seq[:m]) + " ->"
        items.append(
            _item(
                "dyck",
                prompt,
                choices,
                choices.index(gold_tok),
                f"dyck/k{k}/len{n_pairs}",
                pos=m,
            )
        )
    return items


# ------------------------------------------------------------ modular arithmetic


def modular(seed, p=97, ood=False):
    """(a + b) mod p with a held-out operand range (grokking-style ID vs
    OOD split, 2201.02177): ID operands < p//2, OOD operands ≥ p//2."""
    rng = random.Random(int(seed))
    half = p // 2
    lo, hi = (half, p - 1) if ood else (0, half - 1)
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    ans = (a + b) % p
    prompt = f"compute ({a} + {b}) mod {p} . answer:"
    choices, gold = _numeric_choices(rng, ans, spread=9)
    return [
        _item("mod", prompt, choices, gold, f"mod/p{p}" + ("/ood" if ood else "/id"))
    ]


# ------------------------------------------------------------ Knights & Knaves


def knights_knaves(seed, n=3):
    """Small-N K&K (2410.23123-style): statements (single-type claims and
    conjunctions) are generated consistent with the hidden assignment,
    then uniqueness of the solution is verified by brute force over 2^n
    worlds — memorization-proof by construction. Conjunction claims
    break the global-flip symmetry of pure type-claim systems."""
    rng = random.Random(int(seed))
    names = rng.sample(_NAMES, n)
    for _attempt in range(300):
        assign = [rng.random() < 0.5 for _ in range(n)]  # True = knight
        if len(set(assign)) < 2:
            continue
        stmts = []  # (speaker, text, proposition fn over world tuple)
        for i in range(n):
            others = [x for x in range(n) if x != i]
            use_conj = n >= 3 and rng.random() < 0.45
            if use_conj:
                a, b = rng.sample(others, 2)
                if assign[i]:  # knight: conjunction must hold
                    if not (assign[a] and not assign[b]):
                        use_conj = False
                else:  # knave: conjunction must fail
                    if assign[a] and not assign[b]:
                        use_conj = False
            if use_conj:
                text = f"{names[a]} is a knight and {names[b]} is a knave"
                stmts.append((i, text, lambda w, a=a, b=b: w[a] and not w[b]))
            else:
                j = rng.choice(others)
                said = assign[j] if assign[i] else not assign[j]
                word = "knight" if said else "knave"
                stmts.append(
                    (i, f"{names[j]} is a {word}", lambda w, j=j, said=said: w[j] == said)
                )
        sols = [
            w
            for w in itertools.product((False, True), repeat=n)
            if all(w[i] == bool(prop(w)) for i, _t, prop in stmts)
        ]
        if len(sols) == 1 and list(sols[0]) == assign:
            break
    else:
        return []
    text = " ".join(f"{names[i]} says '{t}'." for i, t, _p in stmts)
    qi = rng.randrange(n)
    prompt = (
        "every person is either a knight (always tells the truth) or a knave "
        "(always lies). " + text + f" is {names[qi]} a knight? answer (yes or no):"
    )
    return [
        _item(
            "kk",
            prompt,
            ["yes", "no"],
            0 if assign[qi] else 1,
            f"kk/n{n}",
            answer="yes" if assign[qi] else "no",
        )
    ]
