import tagger as tag

special_ascii = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
special_other = '·\t\n\v\f\r'



sampleline = '사랑하고싶게하는가슴속온감정을헤집어놓는영화예요정말최고 헷 km 1245 #%$&'


# ---------------- #
# ----- File ----- #
# ---------------- #


# file -> lines
def f_to_l(filename):
    return open(filename, 'r', encoding='utf-8').readlines()


# ------------------ #
# ----- String ----- #
# ------------------ #


def s_is_title(line):
    if '<title>' in line:
        return True
    else:
        return False


def s_get_title(line):
    return line.replace('<title>', '').replace('</title>', '')


def s_simplify(line):
    for c in special_ascii:
        line = line.replace(c, ' ')
    for c in special_other:
        line = line.replace(c, '')
    return line


def s_extract(line):
    # drop some characters
    sline = s_simplify(line)

    return tag.extract(sline)





# ----------------- #
# ----- Print ----- #
# ----------------- #


def print_list(list):
    for item in list:
        print(list)

