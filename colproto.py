from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import count
from typing import Any, cast

import awkward as ak
import numba
import numpy as np
from awkward.operations import str as akstr
from pinyinparser import Final, Initial, Syllable, Tone

try:
    from .typing_utils import ArrayLike, AwkwardLike, asArrayLike, asAwkwardLike

    NUMBA_FILE_CACHE = True
except ImportError:
    from typing_utils import ArrayLike, AwkwardLike, asArrayLike, asAwkwardLike

    NUMBA_FILE_CACHE = False  # 防止包内调用的缓存文件把包外调用炸了


class ColProtoABC[T](ABC):
    def __init__(self, data: AwkwardLike) -> None:
        self.data = data

    @staticmethod
    def from_python(raw: list[T]) -> ak.Array:
        return ak.Array(raw)

    @abstractmethod
    def query(self, *args, **kwargs) -> ArrayLike:
        pass

    def find(self, item: T) -> ArrayLike:
        return self.data == item

    data: AwkwardLike


class JITColABC[T](ABC):
    @abstractmethod
    def __init__(self, from_: T) -> None: ...


class PlainText(ColProtoABC[str]):
    def query(self, method: str, args: tuple[Any], kwargs: dict[str, Any]) -> AwkwardLike:
        akstrop = getattr(akstr, method, None)
        if akstrop is None:
            raise ValueError(f"方法 {method} 不存在 @ PlainText")
        return akstrop(self.data, *args, **kwargs)


class _Int(ColProtoABC[int]):
    def query(self, method: str, target: int) -> AwkwardLike:
        match method:
            case "eq":
                return self.data == target
            case "ne":
                return self.data != target
            case "gt":
                return self.data > target
            case "ge":
                return self.data >= target
            case "lt":
                return self.data < target
            case "le":
                return self.data <= target
        raise ValueError(f"方法 {method} 不存在 @ Int")


class Int64(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.int64))


class UInt64(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.uint64))


class Int32(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.int32))


class UInt32(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.uint32))


class Int16(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.int16))


class UInt16(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.uint16))


class Int8(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.int8))


class UInt8(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.uint8))


class Bool(_Int):
    @staticmethod
    def from_python(raw: list[int]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.bool_))


class _Float(ColProtoABC[float]):
    def query(self, method: str, target: float) -> AwkwardLike:
        match method:
            case "eq":
                return self.data == target
            case "ne":
                return self.data != target
            case "gt":
                return self.data > target
            case "ge":
                return self.data >= target
            case "lt":
                return self.data < target
            case "le":
                return self.data <= target
        raise ValueError(f"方法 {method} 不存在 @ Float")


class Float32(_Float):
    @staticmethod
    def from_python(raw: list[float]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.float32))


class Float64(_Float):
    @staticmethod
    def from_python(raw: list[float]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.float64))


class _Coplex(ColProtoABC[complex]):
    def query(self, method: str, target: complex) -> AwkwardLike:
        match method:
            case "eq":
                return self.data == target
            case "ne":
                return self.data != target
        raise ValueError(f"方法 {method} 不存在 @ Complex")


class Complex64(_Coplex):
    @staticmethod
    def from_python(raw: list[complex]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.complex64))


class Complex128(_Coplex):
    @staticmethod
    def from_python(raw: list[complex]) -> ak.Array:
        return ak.Array(np.array(raw, dtype=np.complex128))


class Pinyin(ColProtoABC[list[int]]):
    @staticmethod
    def from_python(raw: list[list[int]]) -> ak.Array:
        data = asAwkwardLike(ak.values_astype(ak.Array(raw), "uint16"))
        offsets = np.asarray(data.layout.offsets)
        max_offset = offsets[-1]
        i32max = np.iinfo(np.int32).max
        if max_offset <= i32max:
            new_offsets = offsets.astype(np.int32)
        else:
            new_offsets = offsets.astype(np.int64)
        new_index = ak.index.Index(new_offsets)
        new_layout = ak.contents.ListOffsetArray(new_index, data.layout.content)
        return ak.Array(new_layout)

    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def __p_check(offsets, data16, m, n, and16, xand16):

        nstring = len(offsets) - 1
        result = np.zeros(nstring, dtype=np.bool_)
        if m is None and n is None:
            n = len(and16)
            for i in numba.prange(nstring):
                if (offsets[i + 1] - offsets[i]) != n:
                    continue
                base = offsets[i]
                for j in range(n):
                    if (data16[base + j] & and16[j]) != xand16[j]:
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
                    if (data16[base + j] & and16[j]) != xand16[j]:
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
                    if (data16[base + j] & and16[j]) != xand16[j]:
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
                    if (data16[base + j] & and16[j]) != xand16[j]:
                        break
                else:
                    result[i] = True
        else:
            raise IndexError
        return result

    __pw_o64 = np.array([0], dtype=np.int64)
    __pw_o32 = np.array([0], dtype=np.int32)
    __pw_d = np.array([], dtype=np.uint16)
    __pw_a = np.array([0], dtype=np.uint16)
    __pw_x = np.array([0], dtype=np.uint16)
    __p_check(__pw_o64, __pw_d, 0, 0, __pw_a, __pw_x)
    __p_check(__pw_o64, __pw_d, -1, -1, __pw_a, __pw_x)
    __p_check(__pw_o64, __pw_d, None, None, __pw_a, __pw_x)
    __p_check(__pw_o64, __pw_d, -1, None, __pw_a, __pw_x)
    __p_check(__pw_o32, __pw_d, 0, 0, __pw_a, __pw_x)
    __p_check(__pw_o32, __pw_d, -1, -1, __pw_a, __pw_x)
    __p_check(__pw_o32, __pw_d, None, None, __pw_a, __pw_x)
    __p_check(__pw_o32, __pw_d, -1, None, __pw_a, __pw_x)

    def query(self, m: int | None, n: int | None, iw: bool, fw: bool, s: list[Syllable]) -> ArrayLike:
        offset: ArrayLike = self.data.layout.offsets.data
        data: ArrayLike = self.data.layout.content.data

        and_lst: list[np.uint16] = []
        xor_lst: list[np.uint16] = []
        for syll in s:
            and_val = 0
            xor_val = int(syll)

            if syll.initial not in (Initial.missing, Initial.unspec):
                and_val |= 0x001F if iw else 0x801F
            if syll.final not in (Final.missing, Final.unspec):
                and_val |= 0x1F00 if fw else 0x7F00
            if syll.tone not in (Tone.missing, Tone.unspec):
                and_val |= 0x00E0

            and_lst.append(np.uint16(and_val))
            xor_lst.append(np.uint16(xor_val))

        and_np = np.array(and_lst, dtype=np.uint16)
        xand_np = np.array(xor_lst, dtype=np.uint16) & and_np

        return asArrayLike(self.__p_check(offset, data, m, n, and_np, xand_np))


class _Enum(_Int):
    @staticmethod
    def _from_python(raw: list[str], dtype) -> ak.Array:
        enummap: defaultdict[str, int] = defaultdict(count(1).__next__)
        data = np.empty(len(raw), dtype=dtype)
        for i, s in enumerate(raw):
            data[i] = enummap[s]
        return ak.with_parameter(ak.Array(data), "enummap", enummap)

    def query(self, method: str, target: str):
        enummap: dict[str, int] = cast(dict[str, int], ak.parameters(self.data)["enummap"])
        match method:
            case "eq":
                if target not in enummap:
                    return ak.Array(np.zeros(len(self.data), dtype=np.bool_))
                return self.data == enummap[target]
            case "ne":
                if target not in enummap:
                    return ak.Array(np.ones(len(self.data), dtype=np.bool_))
                return self.data != enummap[target]
            case _:
                raise ValueError(f"方法 {method} 不存在 @ Enum")


class Enum8(_Enum):
    @staticmethod
    def from_python(raw: list[str]) -> ak.Array:
        return _Enum._from_python(raw, np.uint8)


class Enum16(_Enum):
    @staticmethod
    def from_python(raw: list[str]) -> ak.Array:
        return _Enum._from_python(raw, np.uint16)


class Enum32(_Enum):
    @staticmethod
    def from_python(raw: list[str]) -> ak.Array:
        return _Enum._from_python(raw, np.uint32)


class Enum64(_Enum):
    @staticmethod
    def from_python(raw: list[str]) -> ak.Array:
        return _Enum._from_python(raw, np.uint64)


class _Length(JITColABC[PlainText], _Int):
    @staticmethod
    @abstractmethod
    def __p_len(offset, data):
        pass

    def __init__(self, from_: PlainText) -> None:
        self.data = asAwkwardLike(ak.Array(self.__p_len(from_.data.layout.offsets.data, from_.data.layout.content.data)))

    @staticmethod
    def from_python(*_, **_____):
        return NotImplemented


class Length8(_Length, UInt8):
    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def __p_len(offsets, data):
        nstr = len(offsets) - 1
        lengths = np.empty(nstr, dtype=np.uint8)
        for i in numba.prange(nstr):
            cnt = np.uint8(0)
            for j in range(offsets[i], offsets[i + 1]):
                if (data[j] & 0xC0) != 0x80:
                    cnt += 1
            lengths[i] = cnt
        return lengths

    __p_len(np.array([0], dtype=np.int64), np.array([], dtype=np.uint8))


class Length16(_Length, UInt16):
    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def __p_len(offsets, data):  # DRY不了，numba没法推断这种外源类型，所以只能整个copy一遍，唉太坏了
        nstr = len(offsets) - 1
        lengths = np.empty(nstr, dtype=np.uint16)
        for i in numba.prange(nstr):
            cnt = np.uint16(0)
            for j in range(offsets[i], offsets[i + 1]):
                if (data[j] & 0xC0) != 0x80:
                    cnt += 1
            lengths[i] = cnt
        return lengths

    __p_len(np.array([0], dtype=np.int64), np.array([], dtype=np.uint8))


class Length32(_Length, UInt32):
    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def __p_len(offsets, data):
        nstr = len(offsets) - 1
        lengths = np.empty(nstr, dtype=np.uint32)
        for i in numba.prange(nstr):
            cnt = np.uint32(0)
            for j in range(offsets[i], offsets[i + 1]):
                if (data[j] & 0xC0) != 0x80:
                    cnt += 1
            lengths[i] = cnt
        return lengths

    __p_len(np.array([0], dtype=np.int64), np.array([], dtype=np.uint8))


class Length64(_Length, UInt64):
    @staticmethod
    @numba.njit(parallel=True, cache=NUMBA_FILE_CACHE)
    def __p_len(offsets, data):
        nstr = len(offsets) - 1
        lengths = np.empty(nstr, dtype=np.uint64)
        for i in numba.prange(nstr):
            cnt = np.uint64(0)
            for j in range(offsets[i], offsets[i + 1]):
                if (data[j] & 0xC0) != 0x80:
                    cnt += 1
            lengths[i] = cnt
        return lengths

    __p_len(np.array([0], dtype=np.int64), np.array([], dtype=np.uint8))


PROTO_NAMES: dict[str, type[ColProtoABC]] = {
    "PT": PlainText,
    "B": Bool,
    "U8": UInt8,
    "I8": Int8,
    "U16": UInt16,
    "I16": Int16,
    "U32": UInt32,
    "I32": Int32,
    "U64": UInt64,
    "I64": Int64,
    "F32": Float32,
    "F64": Float64,
    "C64": Complex64,
    "C128": Complex128,
}

JIT_NAMES: dict[str, type[JITColABC]] = {
    "L8": Length8,
    "L16": Length16,
    "L32": Length32,
    "L64": Length64,
}
