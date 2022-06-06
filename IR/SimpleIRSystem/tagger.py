from konlpy.tag import Okt, Hannanum, Kkma

okt = Okt()
kkma = Kkma()

# ------------------------- #
# ----- Select Tagger ----- #
# ------------------------- #
_okt_drop_pos = [ 'Punctuation', 'KoreanParticle', 'Hashtag', 'ScreenName', 'Email', 'URL' ]
_okt_grind_pos = [ 'Number' ]
_hannanum_drop_pos = [ 'JC', 'JX', 'JP', 'EP', 'EF', 'EC', 'ET', 'XP' ]


def pos_okt(txt):
    return okt.pos(txt, norm=True, stem=True, join=False)


def pos_kkma(txt):
    return kkma.pos(txt, flatten=True, join=False)


def extract(txt):
    approved = []
    word_pos = pos_okt(txt)
    for word, pos in word_pos:
        # drop useless pos
        if pos in _okt_drop_pos:
            continue

        # grind certain pos
        if pos in _okt_grind_pos:
            for w, p in pos_kkma(word):
                approved.append(w)

        # good pos
        else:
            approved.append(word)

    return approved

