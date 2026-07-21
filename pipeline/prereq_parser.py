"""
Turns SAP's raw prerequisite tokens into a nested AND/OR tree, so the planner
can evaluate "has this student met the prerequisites for course X" in code
instead of re-parsing a Hebrew display string.

SAP represents a prerequisite expression as a flat list of tokens, each with:
  - bracket: "(", ")", or "" - a paren to emit before this token
  - course:  an 8-digit course number, or "" if this token is just a paren
  - operator: "AND", "OR", or "" - the operator that follows this token

e.g. the expression "(02340114) or (02340117)" arrives as:
  [{bracket: "(", course: "02340114", operator: ""},
   {bracket: ")", course: "",         operator: "OR"},
   {bracket: "(", course: "02340117", operator: ""},
   {bracket: ")", course: "",         operator: ""}]

Tree shape produced:
  {"course": "02340114"}
  {"op": "and" | "or", "args": [tree, tree, ...]}
"""


def _tokenize(prereq_tokens: list[dict]) -> list:
    """Flatten SAP's token list into a simple stream of:
    '(' / ')' / ('course', number) / 'AND' / 'OR'"""
    stream = []
    for token in prereq_tokens:
        for ch in token["bracket"]:
            stream.append(ch)
        if token["course"]:
            stream.append(("course", token["course"]))
        if token["operator"] == "AND":
            stream.append("AND")
        elif token["operator"] == "OR":
            stream.append("OR")
        elif token["operator"]:
            raise ValueError(f"Unknown operator: {token['operator']}")
    return stream


class _Parser:
    """Recursive-descent parser using standard AND-before-OR precedence:
        expr   := conj ("OR" conj)*
        conj   := term ("AND" term)*
        term   := ("course") | "(" expr ")"
    SAP brackets groups when it bothers to, but often leaves a run like
    "A AND B OR A AND C OR A AND D" unbracketed inside one outer paren,
    trusting normal precedence to mean (A AND B) OR (A AND C) OR (A AND D) -
    which is also the natural reading of the equivalent Hebrew sentence."""

    def __init__(self, stream: list):
        self.stream = stream
        self.pos = 0

    def peek(self):
        return self.stream[self.pos] if self.pos < len(self.stream) else None

    def next(self):
        value = self.peek()
        self.pos += 1
        return value

    def parse_expr(self):
        args = [self.parse_conj()]
        while self.peek() == "OR":
            self.next()
            args.append(self.parse_conj())
        return args[0] if len(args) == 1 else {"op": "or", "args": args}

    def parse_conj(self):
        args = [self.parse_term()]
        while self.peek() == "AND":
            self.next()
            args.append(self.parse_term())
        return args[0] if len(args) == 1 else {"op": "and", "args": args}

    def parse_term(self):
        token = self.next()
        if token == "(":
            inner = self.parse_expr()
            if self.next() != ")":
                raise ValueError("Unbalanced parens in prereq expression")
            return inner
        if isinstance(token, tuple) and token[0] == "course":
            return {"course": token[1]}
        raise ValueError(f"Unexpected token: {token}")


def collect_prereq_courses(tree: dict | None) -> list[str]:
    """All course numbers appearing anywhere in a parsed prereq tree,
    regardless of AND/OR structure. Used to build the reverse-prerequisite
    graph (which courses depend on a given one), where the simplification of
    "appears anywhere" vs. exact boolean semantics is an acceptable
    approximation for a "what does failing this course block" heuristic."""
    if tree is None:
        return []

    courses: list[str] = []

    def walk(node: dict):
        if "course" in node:
            courses.append(node["course"])
        else:
            for arg in node["args"]:
                walk(arg)

    walk(tree)
    return courses


def parse_prereq_tree(prereq_tokens: list[dict]) -> dict | None:
    """Returns None if there are no prerequisites, otherwise a nested
    {"course": ...} / {"op": "and"|"or", "args": [...]} tree."""
    if not prereq_tokens:
        return None

    stream = _tokenize(prereq_tokens)
    if not stream:
        return None

    parser = _Parser(stream)
    tree = parser.parse_expr()
    if parser.pos != len(stream):
        raise ValueError(f"Leftover tokens after parsing: {stream[parser.pos:]}")
    return tree
