if __name__ == '__main__':
    print('Main Routine : Index')

import globals as gb
import analyze as an
import dump as dp
import math
from tqdm import tqdm

also_write_in_txt = True

"""
# ------------------------------------- #
# ----- Term Frequency Dictionary ----- #
# ------------------------------------- #
"""
print('begin -> Term Frequency Dictionary')
#   term_dict = {
#       'term' : { doc# : count, ... },
#       ...
#   }
term_dict = dict()

def write_term_dict(doc, doc_count):
    for line in doc:
        words = an.s_extract(line)
        for word in words:

            if word not in term_dict:
                term_dict[word] = dict()
            indices = term_dict[word]

            if doc_count not in indices:
                indices[doc_count] = 0

            indices[doc_count] += 1

flines = an.f_to_l(gb.f_coll)
doc_count = 0
doc = []

for fline in tqdm(flines):
    if an.s_is_title(fline):

        # write-back dictionary
        write_term_dict(doc, doc_count)

        # start new doc
        doc_count += 1
        doc.clear()
        doc.append(an.s_get_title(fline))

    else:
        doc.append(fline)
write_term_dict(doc, doc_count)
# pickle
dp.write(term_dict, gb.f_pkl_term)

print('end')
"""
# ------------------------------------ #
# ----- Log Frequency Dictionary ----- #
# ------------------------------------ #
"""
print('begin -> Log Frequency Dictionary')
#   tf_dict = {
#       'term' : { doc# : log-frequency, ... },
#       ...
#   }
tf_dict = dict()

for term, indices in term_dict.items():
    if term not in tf_dict:
        tf_dict[term] = dict()
    d = tf_dict[term]

    for doc, count in indices.items():
        # w = 1 + log10( tf(t,d) )
        d[doc] = 1 + math.log10(count)

dp.write(tf_dict, gb.f_pkl_tf)
print('end')
"""
# -------------------------------------- #
# ----- Inverse Document Frequency ----- #
# -------------------------------------- #
"""
print('begin -> Inverse Document Frequency')
#   idf_dict = {
#       'term' : idf-weight,
#       ...
#   }

_df_dict = dict()
idf_dict = dict()

# get document-frequency
_df_max = 0
for term, indices in term_dict.items():
    sum = 0
    for doc, count in indices.items():
        sum += count
    _df_dict[term] = sum
    _df_max = max (_df_max, sum)

# get inversed-df
for term, df in _df_dict.items():
    idf_dict[term] = math.log10(_df_max / df)

dp.write(idf_dict, gb.f_pkl_idf)
print('end')

"""
# ------------------------------- #
# ----- tf-idf Weight Table ----- #
# ------------------------------- # 
"""
print('begin -> tf-idf Weight Table')
#   w_table = {
#       'term' : { doc : weight, ...  },
#       ...
#   }

w_table = dict()

for term, tfs in tf_dict.items():
    d = dict()
    idf = idf_dict[term]
    for doc, tf in tfs.items():
        d[doc] = tf * idf
    w_table[term] = d

dp.write(w_table, gb.f_pkl_w)
print('end')

# ------------------------------- #
# ----- Result in txt file? ----- #
# ------------------------------- #

def write_txt(yes):
    if yes:
        dp.pkl_to_txt(gb.f_pkl_term, gb.f_txt_term)
        dp.pkl_to_txt(gb.f_pkl_tf, gb.f_txt_tf)
        dp.pkl_to_txt(gb.f_pkl_idf, gb.f_txt_idf)
        dp.pkl_to_txt(gb.f_pkl_w, gb.f_txt_w)
write_txt(also_write_in_txt)