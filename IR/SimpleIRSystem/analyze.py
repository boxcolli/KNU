from konlpy.tag import Okt

okt = Okt()

stopwordstring = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


sampleline = '사랑하고싶게하는가슴속온감정을헤집어놓는영화예요정말최고 헷 km 1245 #%$&'


# ---- File --------------------------------------- #
def file_to_string(filename):
    return open(filename, 'r', encoding='utf-8').read()


def file_to_lines(filename):
    return open(filename, 'r', encoding='utf-8').readlines()


# ---- String --------------------------------------- #


def str_is_title(line):
    if '<title>' in line:
        return True
    else:
        return False


def str_get_title(line):
    return line.replace('<title>', '').replace('</title>', '')


def str_get_simple_line(line):
    for c in stopwordstring:
        line = line.replace(c, ' ')
    return line

def str_morph_me(txt):
    return okt.morphs(txt, norm=True, stem=True)


def print_list(list):
    for item in list:
        print(list)

