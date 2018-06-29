import random


digits = [
    'ein',
    'zwei',
    'drei',
    'vier',
    'fuenf',
    'sechs',
    'sieben',
    'acht',
    'neun'
    ]
tenners = [
    'zehn',
    'zwanzig',
    'dreissig',
    'vierzig',
    'fuenfzig',
    'sechzig',
    'siebzig',
    'achtzig',
    'neunzig'
    ]
exceptions = {
    1: 'eins',
    11: 'elf',
    12: 'zwoelf',
    13: 'dreizehn',
    14: 'vierzehn',
    15: 'fuenfzehn',
    16: 'sechzehn',
    17: 'siebzehn',
    18: 'achtzehn',
    19: 'neunzehn'
    }


def _to_german(number, with_s=True):
    thousands, thousand_remainder = divmod(number, 1000)
    hundreds, hundred_remainder = divmod(thousand_remainder, 100)
    tens, singles = divmod(hundred_remainder, 10)

    if number > 1e6:
        raise NotImplemented("Numbers above 1000000 are not supported,"
                             " got %i." % number)

    if number == 1e6:
        return ["eine", "millionen"]

    stack = []
    if thousands:
        if thousands > 9:
            stack.extend(_to_german(thousands, with_s=False))
            stack.append('tausend')
        else:
            stack.append(digits[thousands - 1] + 'tausend')
    if hundreds:
        stack.append(digits[hundreds - 1] + 'hundert')

    if hundred_remainder in exceptions:
        if with_s or hundred_remainder != 1:
            if stack:
                return stack + ['und'] + [exceptions[hundred_remainder]]
            else:
                return [exceptions[hundred_remainder]]
        else:
            return stack + ['und', 'ein']

    if singles:
        stack.append(digits[singles - 1])

    if tens:
        stack.append(tenners[tens - 1])
    if len(stack) > 1:
        if not (stack[-1].endswith('hundert') or
                stack[-1].endswith('tausend')):
            return stack[:-1] + ['und'] + [stack[-1]]
        else:
            return stack
    else:
        return stack


def to_german(number):
    return ' '.join(_to_german(number))


def generate(n_samples=100, low=0, high=10000,
             exhaustive=None, seed=1):
    rng = random.Random(seed)
    pairs = {}
    if exhaustive is not None:
        for number in range(1, exhaustive):
            pairs[number] = to_german(number)

    while len(pairs) < n_samples:
        r = rng.randint(low, high)
        if r not in pairs:
            pairs[r] = to_german(r)

    return [str(n) for n in pairs.keys()], list(pairs.values())


if __name__ == "__main__":
    # very poor man's testing
    for digit, text in generate(10).items():
        print(digit, text)

    for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 20, 21, 22,
              31, 35, 41, 100, 101,
              120, 1200, 1201, 301000, 400000, 400001, 512369,
              666, 42, 984561,
              1000000):
        print(i, ' '.join(to_german(i)))
