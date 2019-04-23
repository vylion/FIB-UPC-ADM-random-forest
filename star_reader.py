import csv
from timeit import default_timer as timer
from star import Star

STAR_CLASSES = 'OBAFGKMC'
KEPT_DATA = ['rv', 'absmag', 'ci', 'lum']


def make_star(header, row, fields=None):
    data = {}
    types = []

    for field, value in zip(header, row):
        try:
            num = float(value)
            if num == int(num):
                num = int(num)
            value = num
        except ValueError:
            if value == '':
                value = None

        if field == 'dist' and value >= 100000:
            # Discarding star with dubious value
            return None

        if field == 'spect':
            if value is None:
                # Discarding unclassified star
                return None

            type_list = value.split('/')
            types = []
            for sp_type in type_list:
                if sp_type and sp_type[0] in STAR_CLASSES:
                    types.append(sp_type[0])
            value = ''.join(set(types))

        data[field] = value

    display_name = data['proper'] or data['bf'] or ('ID ' + str(data['id']))
    fields = fields or header

    return Star(data['spect'], display_name, data, fields)


def read_stars(fields=KEPT_DATA):
    print("Parsing stars...")
    star_list = []
    header = None

    t_start = timer()

    with open('hygdata_v3.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

        for row in reader:
            star = make_star(header, row, fields)
            if star is not None:
                star_list.append(star)

    csv_file.close()

    t_end = timer()

    print("Parsed {} stars.\nElapsed time: {:.3f}\n".format(len(star_list), t_end-t_start)) # noqa

    return star_list, fields or header


if __name__ == "__main__":
    read_stars()
