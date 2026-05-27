import pickle
import time
import traceback

import awkward as ak
import numba
import numpy as np
from pinyinparser import Final, Initial, Tone, parse


@numba.njit(parallel=True, cache=True)
def _p_check(offsets, data, m, n, and_bytes, xand_bytes):
    nstring = len(offsets) - 1
    result = np.zeros(nstring, dtype=np.bool_)
    if m is None and n is None:
        n = len(and_bytes)
        for i in numba.prange(nstring):
            if (offsets[i + 1] - offsets[i]) != n:
                continue
            base = offsets[i]
            for j in range(n):
                if (data[base + j] & and_bytes[j]) != xand_bytes[j]:
                    break
            else:
                result[i] = True
    elif m < 0 and n is None:
        width = -m
        for i in numba.prange(nstring):
            if (offsets[i + 1] - offsets[i]) < width:
                continue
            base = offsets[i + 1] + m
            for j in range(width):
                if (data[base + j] & and_bytes[j]) != xand_bytes[j]:
                    break
            else:
                result[i] = True
    elif m is None or n is None:
        raise IndexError
    elif m >= 0 and n >= 0:
        width = n - m
        for i in numba.prange(nstring):
            if (offsets[i + 1] - offsets[i]) < n:
                continue
            base = offsets[i] + m
            for j in range(width):
                if (data[base + j] & and_bytes[j]) != xand_bytes[j]:
                    break
            else:
                result[i] = True
    elif m < 0 and n < 0:
        width = n - m
        abs_m = -m
        for i in numba.prange(nstring):
            if (offsets[i + 1] - offsets[i]) < abs_m:
                continue
            base = offsets[i + 1] + m
            for j in range(width):
                if (data[base + j] & and_bytes[j]) != xand_bytes[j]:
                    break
            else:
                result[i] = True
    else:
        raise IndexError
    return result


def _awk2buf(s):
    layout = ak.to_layout(s)
    offsets = layout.offsets.data
    data = layout.content.data
    return offsets, data


def _mask2np(AND, XOR):
    AND_np = np.frombuffer(AND, dtype=np.uint8).copy()
    XAND_np = np.frombuffer(XOR, dtype=np.uint8) & AND_np
    assert len(AND_np) == len(XAND_np)
    return AND_np, XAND_np


def find(s, m, n, AND, XOR):
    offsets, data = _awk2buf(s)
    AND_np, XAND_np = _mask2np(AND, XOR)
    return _p_check(offsets, data, m, n, AND_np, XAND_np)


def warmup():
    oi64 = np.array([0], dtype=np.int64)
    ou32 = np.array([0], dtype=np.uint32)
    d = np.array([], dtype=np.uint8)
    a = np.array([0], dtype=np.uint8)
    x = np.array([0], dtype=np.uint8)
    _p_check(oi64, d, 0, 0, a, x)
    _p_check(oi64, d, -1, -1, a, x)
    _p_check(oi64, d, None, None, a, x)
    _p_check(oi64, d, -1, None, a, x)
    _p_check(ou32, d, 0, 0, a, x)
    _p_check(ou32, d, -1, -1, a, x)
    _p_check(ou32, d, None, None, a, x)
    _p_check(ou32, d, -1, None, a, x)


def build_masks(syllables, mask_i=False, mask_f=False):
    and_bytes_list = []
    xor_bytes_list = []
    for syl in syllables:
        and_val = 0
        xor_val = int(syl)
        if syl.initial not in (Initial.missing, Initial.unspec):
            # 若 mask_i 为真，则屏蔽声母变体选择位 0x8000，AND掩码仅保留 0x001F
            and_val |= 0x001F if mask_i else 0x801F
        if syl.tone not in (Tone.missing, Tone.unspec):
            and_val |= 0x00E0
        if syl.final not in (Final.missing, Final.unspec):
            # 若 mask_f 为真，则屏蔽韵母变体选择位 0x6000，AND掩码仅保留 0x1F00
            and_val |= 0x1F00 if mask_f else 0x7F00
        and_bytes_list.append(and_val.to_bytes(2, "little"))
        xor_bytes_list.append(xor_val.to_bytes(2, "little"))
    return b"".join(and_bytes_list), b"".join(xor_bytes_list)


t0 = time.perf_counter()
warmup()
print(f"numba预热耗时: {time.perf_counter() - t0:.3f} 秒")
# print("----- asm -----")
# print(_p_check.inspect_asm())
# print("----- llvm -----")
# print(_p_check.inspect_llvm())
# print("----- end -----")


def load_data(filepath):
    print(f"加载 {filepath} ...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    pinyin_col, word_col = None, None
    pinyin_col = data["pinyin"]
    word_col = data["word"]
    if pinyin_col is None or word_col is None:
        raise ValueError
    print("就绪")
    return pinyin_col, word_col


def main():
    try:
        pinyin_col, word_col = load_data(r"T:\词.awk")
    except Exception as e:
        print(f"加载失败: {e}")
        return
    t0 = time.perf_counter()
    offsets, data = _awk2buf(pinyin_col)
    print(f"awk预处理耗时: {time.perf_counter() - t0:.3f} 秒")
    while True:
        user_input = input("\n> ").strip()
        try:
            # 修改为允许解析5个参数
            parts = [p.strip() for p in user_input.split(",", 4)]
            if len(parts) != 5:
                print("E1")
                continue
            m_syll = int(parts[0]) if parts[0] else None
            n_syll = int(parts[1]) if parts[1] else None
            # 空或空字符视为假，其他任何值视为真
            mask_i = bool(parts[2])
            mask_f = bool(parts[3])
            pinyin_str = parts[4]

            syllables = parse(pinyin_str)
            AND, XOR = build_masks(syllables, mask_i, mask_f)

            m_byte = m_syll and m_syll * 2
            n_byte = n_syll and n_syll * 2
            tm0 = time.perf_counter()
            AND_np, XAND_np = _mask2np(AND, XOR)
            mask = _p_check(offsets, data, m_byte, n_byte, AND_np, XAND_np)
            print(f"耗时: {time.perf_counter() - tm0:.6f} 秒")
            matched_indices = np.where(mask)[0]
            total_found = len(matched_indices)
            if total_found == 0:
                print("<无匹配项>")
                continue
            display_count = min(total_found, 64)
            print(f"{display_count}/{total_found}：")
            for i in range(display_count):
                idx = matched_indices[i]
                print(word_col[idx])
        except ValueError as ve:
            print(f"解析错误: {ve}")
        except Exception as e:
            print(f"运行错误: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
