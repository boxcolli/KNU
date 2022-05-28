if __name__ == '__main__':
    print('PyCharm')

import os

print('using:', os.environ.get('JAVA_HOME'))

import analyze as an
from tqdm import tqdm

filename = 'example.txt'
term_dict = dict()


single = '<title>1. 지미 카터</title>\
지미 카터는 민주당 출신 미국 39번째 대통령이다.\
지미 카터는 조지아 주  한 마을에서 태어났다. \
조지아 공과대학교를 졸업하였다. \
그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다.'

double = '<title>1. 지미 카터</title>\
지미 카터는 민주당 출신 미국 39번째 대통령이다.\
지미 카터는 조지아 주  한 마을에서 태어났다. \
조지아 공과대학교를 졸업하였다. \
그 후 해군에 들어가 전함·원자력·잠수함의 승무원으로 일하였다.\
\
<title>2. 체첸 공화국</title>\
체첸 공화국 또는 줄여서 체첸은 러시아의 공화국이다. \
체첸에서 사용되는 언어는 체첸어와 러시아어이다. \
체첸어는 캅카스제어 중, 북동 캅카스제어로 불리는 그룹에 속하는데 인구시어와 매우 밀접한 관계에 있다.'





def push_index(str, doc_i):
    if str == '':
        return

    # make index of this string
    for stem in an.str_morph_me(an.str_get_simple_line(str)): # get stems
        # new entry?
        if stem not in term_dict:
            term_dict[stem] = dict()

        term_indices = term_dict[stem]

        # new doc entry?
        if doc_i not in term_indices:
            term_indices[doc_i] = 1
        else:
            term_indices[doc_i] += 1


lines = an.file_to_lines(filename)
buffer = ''
doc_num = 0


with tqdm(total=2) as pbar:
    for line in lines:
        if an.str_is_title(line):
            # ---- Tag and Insert ----#
            push_index(buffer, doc_num)
            buffer = an.str_get_title(line)
            doc_num += 1
            pbar.update(1)
        else:
            buffer += line
            # ---- Split Document ----#
    push_index(buffer, doc_num)
    doc_num += 1

print(term_dict)
for term, index in term_dict.items():
    print(f'indices of [{term}]')
    for doc, count in index.items():
        print('\t', doc, count)



#---- Count Term ----#
#----