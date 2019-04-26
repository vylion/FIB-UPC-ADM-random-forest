class Star(object):
    def __init__(self, label, display_name, data, fields):
        self.label = label
        self.name = display_name
        data_list = []
        for field in fields:
            data_list.append(data[field])
        self.data = data_list

    def __str__(self):
        if len(self.label) > 1:
            classification = ' or '.join(self.label)
        else:
            classification = self.label

        s = "Star {} {} of spectral type {}"
        return s.format(self.name, self.data, classification)
