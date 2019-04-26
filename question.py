
def is_numeric(value):
    # Test if a value is numeric
    return isinstance(value, int) or isinstance(value, float)


class Question(object):
    def __init__(self, fields, pos, value):
        self.fields = fields
        self.pos = pos
        self.value = value
        self.numeric = is_numeric(value)

    def match(self, entry):
        val = entry.data[self.pos]

        if self.numeric:
            return val and val > self.value
        else:
            return val == self.value

    def __str__(self):
        condition = self.numeric and ">" or "="
        field = self.fields[self.pos]

        return "Is {f} {cond} {val}?".format(f=field, cond=condition, val=self.value)
