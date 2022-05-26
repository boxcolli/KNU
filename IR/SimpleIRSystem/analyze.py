from konlpy.tag import Okt
okt = Okt()

def fileToString(filename):
    return open(filename, 'r', encoding='utf-8').read()

def tagMe(txt):
    return okt.pos(txt, norm=True, stem=True, join=True)
