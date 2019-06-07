'''
Parse tree utilities
'''

import re

from utils import pairwise


# Because some of the parse trees can be quite deep, make sure to use looping rather than recursion
# for all operations, as it's possible to hit the Python max-recursion depth for these trees.
class ParseTree(object):
    ''' An object that encapsulates a parse tree '''
    CONSTITUENT_REGEX = re.compile(r'<([^\d]+)(\d+)>')
    PARSE_REGEX = re.compile(r'([^\s)]+)|[^ )]+|\)')
    ROOT_REGEX = re.compile(r'^\(ROOT (.*)\)$')

    def __init__(self, label, children=tuple()):
        ''' Initialize the parse tree '''
        self.label = label
        self.children = list(children)

    @property
    def leaves(self):
        ''' Get the leaf nodes of the tree '''
        leaves = []
        stack = self.children[::-1]
        while stack:
            child = stack.pop()
            if isinstance(child, ParseTree):
                stack += child.children[::-1]
            else:
                leaves.append(child)

        return leaves

    @property
    def subtrees(self):
        ''' Get all the subtrees '''
        return [child for child in self.children if isinstance(child, ParseTree)]

    @property
    def width(self):
        ''' The width of the tree '''
        if self.subtrees:
            width = 0
            stack = self.subtrees
            while stack:
                tree = stack.pop()
                subtrees = tree.subtrees
                if subtrees:
                    stack += subtrees
                else:
                    width += 1

            return width
        else:
            return 1

    def to_latex_string(self):
        ''' Return a latex string representation '''
        depth = 0
        strings = []
        stack = [depth, self]
        while stack:
            context = stack.pop()
            if isinstance(context, ParseTree):
                strings.append(' ' * 2 * depth + f'{context.label}')
                stack += [depth] + context.children[::-1]
                depth += 1
            elif isinstance(context, str):
                strings.append(' ' * 2 * depth + f'{context}')
            else:
                depth = context

        return '\n'.join(strings)

    def to_parenthesized_string(self):
        ''' Return a parenthesized representation '''
        strings = []
        sentinel = 0 # use 0 as a sentinel to denote a closing parenthesis
        stack = [sentinel, self]
        while stack:
            context = stack.pop()
            if isinstance(context, ParseTree):
                strings.append(f'({context.label}')
                stack += [sentinel] + context.children[::-1]
            elif isinstance(context, str):
                strings.append(f'{context}')
            else:
                strings[-1] = strings[-1] + ')'

        return ' '.join(strings)

    def __repr__(self):
        ''' Return a string representation of the tree '''
        return self.to_parenthesized_string()

    def add_child(self, string, splitter=None):
        ''' Add the given child to a node, possibly splitting it '''
        if splitter:
            self.children.extend(splitter(string))
        else:
            self.children.append(string)

    @classmethod
    def from_parenthesized_string(cls, string, splitter=None):
        ''' Generate a ParseTree from a string '''
        if not string:
            return ParseTree('')

        nodes = []
        tree = None

        # Filter out (ROOT ...), which CoreNLP seems to output
        match = cls.ROOT_REGEX.match(string)
        if match:
            string = match[1]

        for match in cls.PARSE_REGEX.finditer(string):
            token = match.group()
            if len(token) > 1 and token.startswith('('):
                nodes.append(cls(token[1:], []))
            elif token == ')' and nodes[-1].children:
                tree = nodes.pop()
                if nodes:
                    nodes[-1].children.append(tree)
            else:
                nodes[-1].add_child(token, splitter)

        return tree

    @classmethod
    def from_latex_string(cls, string, splitter=None):
        ''' Generate a ParseTree from a string '''
        if not string:
            return ParseTree('')

        tree = None
        node = None
        parents = []
        next_token = ''

        tokens = string.split(' ')
        if tokens[0] == '0' and tokens[1] == 'ROOT':
            # Ignore the ROOT node
            tokens = tokens[2:]

        for (depth, token), (next_depth, next_token) in pairwise(zip(tokens[::2], tokens[1::2])):
            depth = int(depth)
            next_depth = int(next_depth)

            if next_depth > depth:
                if not tree:
                    # Start of the tree and active node
                    tree = node = cls(token, [])
                else:
                    # Push onto the stack
                    parents.append(node)

                    # Add new internal node and make it the active node
                    node.children.append(cls(token, []))
                    node = node.children[-1]
            elif next_depth < depth:
                # Add the current token
                node.add_child(token, splitter)

                # Then pop back up the stack, to the currently active node
                for _ in range(depth - next_depth):
                    node = parents.pop()
            else:
                # Same depth, so just add the current token
                node.add_child(token, splitter)

        # Add the final token
        node.add_child(next_token, splitter)
        return tree

    @classmethod
    def from_string(cls, string, splitter=None):
        ''' Generate a ParseTree from a string '''
        # Automatically detect whether it's a parenthesized string or not
        if string.startswith('('):
            # Parenthesized string for backwards compatibility. It's deprecated since it's delimited
            # by parentheses, which breaks down with things like emoticons
            return cls.from_parenthesized_string(string, splitter)
        else:
            return cls.from_latex_string(string, splitter)

    def segment(self, max_leaves=1):
        ''' Segment the parse tree with a maximum number of leaves '''
        segments = []
        stack = [self]
        while stack:
            tree = stack.pop()
            leaves = tree.leaves
            num_leaves = len(leaves)
            if tree.width == 1 or num_leaves <= max_leaves:
                segments.append(f'<{tree.label}{num_leaves}>')
                segments.extend(leaves)
            else:
                # Continue processing the children in order. Since we pop from the back, we want to
                # reverse the list of children and add them to the back of the existing stack.
                stack.extend(child for child in tree.children[::-1] if isinstance(child, ParseTree))

        return segments
