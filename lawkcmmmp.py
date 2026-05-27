import os
from functools import cache

import awkward as ak
import pinyinparser
import pypinyin
from pypinyin_dict.phrase_pinyin_data import cc_cedict, large_pinyin

from colproto import Pinyin, PlainText

cc_cedict.load()
large_pinyin.load()


ppc = cache(pinyinparser.parse)


def compile_lexicon_to_awkward(input_path: str, output_path: str):
    pinyin_int_list = []
    word_list = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if not word:
                continue

            pinyin_results = pypinyin.pinyin(word, style=pypinyin.Style.TONE, heteronym=False)

            syllables_for_word: list[int] = []
            for p_list in pinyin_results:
                p_str = p_list[0]

                if not p_str:
                    syls = [0]
                else:
                    try:
                        syls = [int(s) for s in ppc(p_str, default_tone_neutral=True, force_valid_syllable=True, missing_as_nul=True)]
                    except ValueError:
                        syls = [int(pinyinparser.Syllable(pinyinparser.Initial.missing, pinyinparser.Final.missing, pinyinparser.Tone.missing))]

                syllables_for_word.extend(syls)

            pinyin_int_list.append(syllables_for_word)
            word_list.append(word)

    pinyin_col = Pinyin.from_python(pinyin_int_list)

    print("ok 1")

    awk_array = ak.Array({"pinyin.PY": pinyin_col.data, "word.PT": PlainText.from_python(word_list).data})

    print("ok 2")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ak.to_parquet(awk_array, output_path)

    print("ok 3")


if __name__ == "__main__":
    INPUT_FILE = r"R:\Aha\modules\tianzi\lexicons\Lexicon\词.1l1w"
    OUTPUT_FILE = r"R:\\lexloader\\lexicons\\词.pq"

    compile_lexicon_to_awkward(INPUT_FILE, OUTPUT_FILE)
