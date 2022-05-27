from konlpy.tag import Okt
okt = Okt()


stopwordstring = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def file_to_string(filename):
    return open(filename, 'r', encoding='utf-8').read()


def is_title(line):
    if '<title>' in line:
        return True
    else:
        return False


def get_title(line):
    return line.replace('<title>', '').replace('</title>', '')


def get_simple_line(line):
    for c in stopwordstring:
        line = line.replace(c, ' ')
    return line


def


def tag_me(txt):
    return okt.pos(txt, norm=True, stem=True, join=True)


def print_tags(txt):
    for item in tag_me(txt):
        print(item)