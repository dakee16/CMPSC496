"""HW3 - Stacks and Calculators (11 problems)

Upload this file on the Instructor page (teacher.html -> "Prepare an
assignment"). Converted from the CMPSC 132 HW3 handout.

Two things differ from a flat assignment file, and both are deliberate:

  * The problems are METHODS of a class, not free functions. Each class becomes
    one problem group and each named method becomes one problem; the rest of
    the file travels with it as context, so `Calculator._getPostfix` still has
    `Stack` in scope exactly as written here.

  * The `>>>` examples are NOT decoration. A stateful method like `push` has no
    return value to compare, so the recorded call sequence in its docstring is
    what the oracle is seeded from - `x.push(2); x.push(4); x.peek()` is one
    sequence with one list of results. Editing an example changes the tests.

The `# --- steps: ... ---` line inside each class names the methods that are
exercises. Without it every non-scaffolding method is taken, which would turn
given code like `setExpr` into a problem students are asked to write.
"""


class Node:
    """A single link in a stack. Given to students; not an exercise itself."""

    def __init__(self, value):
        self.value = value
        self.next = None

    def __str__(self):
        return "Node({})".format(self.value)

    __repr__ = __str__


# =========================== Part I - Stack ===========================

class Stack:
    '''A last-in-first-out stack built from linked Node objects.

        >>> x=Stack()
        >>> x.pop()
        >>> x.push(2)
        >>> x.push(4)
        >>> x.push(6)
        >>> x
        Top:Node(6)
        Stack:
        6
        4
        2
        >>> x.pop()
        6
        >>> x
        Top:Node(4)
        Stack:
        4
        2
        >>> len(x)
        2
        >>> x.peek()
        4
    '''
    # --- steps: isEmpty, __len__, push, pop, peek ---

    def __init__(self):
        self.top = None

    def __str__(self):
        temp = self.top
        out = []
        while temp:
            out.append(str(temp.value))
            temp = temp.next
        out = '\n'.join(out)
        return ('Top:{}\nStack:\n{}'.format(self.top, out))

    __repr__ = __str__

    def isEmpty(self):
        '''Return True when the stack holds no items, and False otherwise.

        The stack is empty exactly when there is no top node.

            >>> x = Stack()
            >>> x.isEmpty()
            True
            >>> x.push(5)
            >>> x.isEmpty()
            False
        '''
        return self.top is None

    def __len__(self):
        '''Return how many items are on the stack, so that len(x) works.

        Walk the chain of nodes from the top down, counting as you go. An
        empty stack has length 0.

            >>> x = Stack()
            >>> len(x)
            0
            >>> x.push(2)
            >>> x.push(4)
            >>> x.push(6)
            >>> len(x)
            3
        '''
        count = 0
        temp = self.top
        while temp:
            count += 1
            temp = temp.next
        return count

    def push(self, value):
        '''Add value to the top of the stack.

        Wrap the value in a Node, link that node to the current top, and make
        it the new top. Returns nothing - the effect is the state it leaves
        behind, which the next peek or pop observes.

            >>> x = Stack()
            >>> x.push(2)
            >>> x.push(4)
            >>> x.peek()
            4
            >>> len(x)
            2
        '''
        node = Node(value)
        node.next = self.top
        self.top = node

    def pop(self):
        '''Remove the top item and return its value.

        Return None when the stack is empty - popping an empty stack is not an
        error here. The node that leaves the stack must be unlinked from it.

            >>> x = Stack()
            >>> x.pop()
            >>> x.push(2)
            >>> x.push(4)
            >>> x.pop()
            4
            >>> x.pop()
            2
            >>> x.pop()
        '''
        if self.top is None:
            return None
        node = self.top
        self.top = node.next
        node.next = None
        return node.value

    def peek(self):
        '''Return the value on top of the stack WITHOUT removing it.

        Return None when the stack is empty. After a peek the stack must be
        unchanged, which is the whole difference between peek and pop.

            >>> x = Stack()
            >>> x.peek()
            >>> x.push(2)
            >>> x.push(4)
            >>> x.peek()
            4
            >>> len(x)
            2
        '''
        if self.top is None:
            return None
        return self.top.value


# ======================= Part II - Calculator =========================

class Calculator:
    '''An infix calculator that works by converting to postfix first.'''
    # --- steps: _isNumber, _getPostfix, calculate ---

    def __init__(self):
        self.__expr = None

    @property
    def getExpr(self):
        return self.__expr

    def setExpr(self, new_expr):
        if isinstance(new_expr, str):
            self.__expr = new_expr
        else:
            print('setExpr error: Invalid expression')
            return None

    def _isNumber(self, txt):
        '''Return True when txt holds exactly one number, and False otherwise.

        Surrounding spaces are allowed and ignored. Anything else - two numbers
        separated by a space, or a stray letter - is not a number.

            >>> x=Calculator()
            >>> x._isNumber(' 2.560 ')
            True
            >>> x._isNumber('7 56')
            False
            >>> x._isNumber('2.56p')
            False
        '''
        try:
            float(txt)
            return True
        except (ValueError, TypeError):
            return False

    def _getPostfix(self, txt):
        '''Convert the infix expression txt into postfix, or return None when
        the expression is invalid.

        Every number in the output is written as a float, separated by single
        spaces. Operator precedence is ^ above * and / above + and -, and ^
        associates to the RIGHT, so 7^2^3 means 7^(2^3). Parentheses override
        precedence and must be balanced.

        A leading minus attaches to the number it precedes when an operand is
        expected, so "5 +-3" is five plus negative three.

        An expression is invalid when two numbers sit side by side, when an
        operator is missing an operand, when a bracket is unmatched, or when
        any character other than a digit, a dot, an operator or a bracket
        appears. Return None in every one of those cases.

        Required: _getPostfix must create and use a Stack for expression
        processing.

            >>> x=Calculator()
            >>> x._getPostfix('     2 ^       4')
            '2.0 4.0 ^'
            >>> x._getPostfix('          2 ')
            '2.0'
            >>> x._getPostfix('2.1        * 5        + 3       ^ 2 +         1 +             4.45')
            '2.1 5.0 * 3.0 2.0 ^ + 1.0 + 4.45 +'
            >>> x._getPostfix('2*5.34+3^2+1+4')
            '2.0 5.34 * 3.0 2.0 ^ + 1.0 + 4.0 +'
            >>> x._getPostfix('( .5 )')
            '0.5'
            >>> x._getPostfix ('( ( 2 ) )')
            '2.0'
            >>> x._getPostfix ('2 * (           ( 5 +-3 ) ^ 2 + (1 + 4 ))')
            '2.0 5.0 -3.0 + 2.0 ^ 1.0 4.0 + + *'
            >>> x._getPostfix('2* (       -5 + 3 ) ^2+ ( 1 +4 )')
            '2.0 -5.0 3.0 + 2.0 ^ * 1.0 4.0 + +'
            >>> x._getPostfix('2 * 5 + 3 ^ + -2 + 1 + 4')
            >>> x._getPostfix('2    5')
            >>> x._getPostfix('25 +')
            >>> x._getPostfix(' 2 * ( 5      + 3 ) ^ 2 + ( 1 +4 ')
            >>> x._getPostfix('2 *      5% + 3       ^ + -2 +1 +4')
        '''
        postfixStack = Stack()   # method must use postfixStack
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

        # ── tokenize ──
        tokens = []
        i = 0
        prev = None
        while i < len(txt):
            ch = txt[i]
            if ch.isspace():
                i += 1
                continue
            if ch in '()':
                tokens.append(ch)
                prev = ch
                i += 1
                continue
            if ch in '+-*/^':
                # A sign is part of the NUMBER when an operand is expected.
                if ch in '+-' and (prev is None or prev in '(+-*/^'):
                    j = i + 1
                    while j < len(txt) and (txt[j].isdigit() or txt[j] == '.'):
                        j += 1
                    if j == i + 1:
                        return None          # a sign with no number after it
                    tokens.append(ch + txt[i + 1:j])
                    prev = 'num'
                    i = j
                    continue
                tokens.append(ch)
                prev = ch
                i += 1
                continue
            if ch.isdigit() or ch == '.':
                j = i
                while j < len(txt) and (txt[j].isdigit() or txt[j] == '.'):
                    j += 1
                tokens.append(txt[i:j])
                prev = 'num'
                i = j
                continue
            return None                      # letters, %, anything else
        if not tokens:
            return None

        # ── validate: operands and operators must alternate, brackets balance ──
        expect_operand = True
        depth = 0
        for tok in tokens:
            if tok == '(':
                if not expect_operand:
                    return None
                depth += 1
            elif tok == ')':
                if expect_operand:
                    return None
                depth -= 1
                if depth < 0:
                    return None
            elif tok in precedence:
                if expect_operand:
                    return None
                expect_operand = True
            else:
                if not expect_operand:
                    return None
                try:
                    float(tok)
                except ValueError:
                    return None
                expect_operand = False
        if expect_operand or depth != 0:
            return None

        # ── shunting-yard ──
        output = []
        for tok in tokens:
            if tok == '(':
                postfixStack.push(tok)
            elif tok == ')':
                while not postfixStack.isEmpty() and postfixStack.peek() != '(':
                    output.append(postfixStack.pop())
                postfixStack.pop()                   # discard the '('
            elif tok in precedence:
                while (not postfixStack.isEmpty()
                       and postfixStack.peek() != '('
                       and (precedence[postfixStack.peek()] > precedence[tok]
                            or (precedence[postfixStack.peek()] == precedence[tok]
                                and tok != '^'))):   # ^ is right-associative
                    output.append(postfixStack.pop())
                postfixStack.push(tok)
            else:
                output.append(str(float(tok)))
        while not postfixStack.isEmpty():
            output.append(postfixStack.pop())
        return ' '.join(output)

    @property
    def calculate(self):
        '''Evaluate the stored expression and return the result as a float, or
        None when the expression is invalid.

        Convert to postfix first, then walk the postfix left to right using a
        stack: push each number, and on each operator pop the two most recent
        values, combine them, and push the answer back. The first value popped
        is the RIGHT operand, which is what makes 2-3 give -1 rather than 1.

        Return None on a division by zero as well as on an invalid expression.

        calculate must call _getPostfix, and must use calcStack.

            >>> x=Calculator()
            >>> x.setExpr('4        + 3 -       2')
            >>> x.calculate
            5.0
            >>> x.setExpr('-2 +          3.5')
            >>> x.calculate
            1.5
            >>> x.setExpr('2-3*4')
            >>> x.calculate
            -10.0
            >>> x.setExpr('7^2^3')
            >>> x.calculate
            5764801.0
            >>> x.setExpr(' 3 * ((( 10 - 2*3 )) )')
            >>> x.calculate
            12.0
            >>> x.setExpr('      8 / 4 * (3 - 2.45 * ( 4   - 2 ^ 3 )       ) + 3')
            >>> x.calculate
            28.6
            >>> x.setExpr('2 * ( 4 +        2 * (         5 - 3 ^ 2 ) + 1 ) + 4')
            >>> x.calculate
            -2.0
            >>> x.setExpr(" 4 ++ 3+ 2")
            >>> x.calculate
            >>> x.setExpr("4  3 +2")
            >>> x.calculate
            >>> x.setExpr('( 2 ) * 10 - 3 *( 2 - 3 * 2 ) )')
            >>> x.calculate
        '''
        if not isinstance(self.__expr, str) or len(self.__expr) <= 0:
            print("Argument error in calculate")
            return None

        calcStack = Stack()   # method must use calcStack

        postfix = self._getPostfix(self.__expr)
        if postfix is None:
            return None

        for tok in postfix.split():
            if tok in ('+', '-', '*', '/', '^'):
                if len(calcStack) < 2:
                    return None
                right = calcStack.pop()
                left = calcStack.pop()
                if tok == '+':
                    calcStack.push(left + right)
                elif tok == '-':
                    calcStack.push(left - right)
                elif tok == '*':
                    calcStack.push(left * right)
                elif tok == '/':
                    if right == 0:
                        return None
                    calcStack.push(left / right)
                else:
                    calcStack.push(left ** right)
            else:
                calcStack.push(float(tok))

        if len(calcStack) != 1:
            return None
        return calcStack.pop()


# =================== Part III - AdvancedCalculator ====================

class AdvancedCalculator:
    '''A calculator for several semicolon-separated statements that assign to
    variables and finish with a return.

        >>> C = AdvancedCalculator()
        >>> C.setExpression('a = 5;b = 7 + a;a = 7;c = a + b;c = a * 0;return c')
        >>> C.calculateExpressions() == {'a = 5': {'a': 5.0}, 'b = 7 + a': {'a': 5.0, 'b': 12.0}, 'a = 7': {'a': 7.0, 'b': 12.0}, 'c = a + b': {'a': 7.0, 'b': 12.0, 'c': 19.0}, 'c = a * 0': {'a': 7.0, 'b': 12.0, 'c': 0.0}, '_return_': 0.0}
        True
    '''
    # --- steps: _isVariable, _replaceVariables, calculateExpressions ---

    def __init__(self):
        self.expressions = ''
        self.states = {}

    def setExpression(self, expression):
        self.expressions = expression
        self.states = {}

    def _isVariable(self, word):
        '''Return True when word is a valid variable name, and False otherwise.

        A valid name starts with a letter and contains only letters and digits.

            >>> C = AdvancedCalculator()
            >>> C._isVariable('volume')
            True
            >>> C._isVariable('4volume')
            False
            >>> C._isVariable('volume2')
            True
            >>> C._isVariable('vol%2')
            False
        '''
        return bool(word) and word[0].isalpha() and word.isalnum()

    def _replaceVariables(self, expr):
        '''Return expr with every variable replaced by its value from states,
        or None when the expression cannot be resolved.

        The expression arrives as space-separated words. A word is either an
        operator, a bracket, a number, or a variable name. Replace each
        variable with its current value; return None the moment a variable is
        not in states, or a word is neither a number nor a valid name.

        Numbers pass through exactly as written - '1' stays '1', not '1.0'.

            >>> C = AdvancedCalculator()
            >>> C._replaceVariables('1')
            '1'
            >>> C._replaceVariables('2 + 3 * ( 4 )')
            '2 + 3 * ( 4 )'
            >>> C._replaceVariables('105 + x')
            >>> C._replaceVariables('4bad + 1')
            >>> C.states = {'x1': 23.0, 'x2': 28.0}
            >>> C._replaceVariables('7 * ( x1 - 1 )')
            '7 * ( 23.0 - 1 )'
            >>> C._replaceVariables('x2 - x1')
            '28.0 - 23.0'
        '''
        out = []
        for word in expr.split():
            if word in ('+', '-', '*', '/', '^', '(', ')'):
                out.append(word)
            elif self._isVariable(word):
                if word not in self.states:
                    return None
                out.append(str(self.states[word]))
            else:
                try:
                    float(word)
                except ValueError:
                    return None
                out.append(word)
        return ' '.join(out)

    def calculateExpressions(self):
        '''Run every statement in order and return a report dictionary, or None
        if any statement is invalid.

        Statements are separated by semicolons. Each one is either an
        assignment `name = expression` or the final `return expression`.

        For each assignment, substitute the known variables, evaluate it with a
        Calculator, and store the result in states. The report maps that
        statement's original text to a COPY of states as it stood just after
        that statement ran - a copy, because states keeps changing afterwards.
        The final return value is stored under the key '_return_'.

        If anything is invalid - a bad variable name, an unknown variable, an
        expression that will not evaluate - reset states to an empty dictionary
        and return None.

        calculateExpressions must use calcObj to compute each expression.

            >>> C = AdvancedCalculator()
            >>> C.setExpression('x1 = 5;x2 = 7 * ( x1 - 1 );x1 = x2 - x1;return x2 + x1 ^ 3')
            >>> C.calculateExpressions() == {'x1 = 5': {'x1': 5.0}, 'x2 = 7 * ( x1 - 1 )': {'x1': 5.0, 'x2': 28.0}, 'x1 = x2 - x1': {'x1': 23.0, 'x2': 28.0}, '_return_': 12195.0}
            True
            >>> C.setExpression('A = 1;B = A + 9;2C = A + B;A = 20;D = A + B + C;return D + A')
            >>> C.calculateExpressions() is None
            True
            >>> C.states == {}
            True
        '''
        self.states = {}
        calcObj = Calculator()   # method must use calcObj
        report = {}

        for statement in self.expressions.split(';'):
            statement = statement.strip()
            if not statement:
                continue

            if statement.startswith('return'):
                replaced = self._replaceVariables(statement[len('return'):].strip())
                if replaced is None:
                    self.states = {}
                    return None
                calcObj.setExpr(replaced)
                value = calcObj.calculate
                if value is None:
                    self.states = {}
                    return None
                report['_return_'] = value
                return report

            if '=' not in statement:
                self.states = {}
                return None
            name, expr = statement.split('=', 1)
            name = name.strip()
            if not self._isVariable(name):
                self.states = {}
                return None
            replaced = self._replaceVariables(expr.strip())
            if replaced is None:
                self.states = {}
                return None
            calcObj.setExpr(replaced)
            value = calcObj.calculate
            if value is None:
                self.states = {}
                return None
            self.states[name] = value
            # A COPY: states keeps changing, and the report must remember this
            # statement's snapshot rather than alias the live dictionary.
            report[statement] = dict(self.states)

        self.states = {}
        return None      # no return statement was ever reached
