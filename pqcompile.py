import csv
import sys
from pathlib import Path

import awkward as ak
from pinyinparser import parse as pinyin_parse

try:
    from .colproto import (
        PROTO_NAMES,
        Bool,
        Pinyin,
        PlainText,
        _Complex,
        _Enum,
        _Float,
        _Int,
    )
except ImportError:
    from colproto import (
        PROTO_NAMES,
        Bool,
        Pinyin,
        PlainText,
        _Complex,
        _Enum,
        _Float,
        _Int,
    )


def compile_csv_tsv(filepath: str | Path, delimiter: str = ",") -> ak.Array:
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames
        if not headers:
            raise ValueError("文件为空或缺少表头")

        raw_data = {h: [] for h in headers}
        for row in reader:
            for h in headers:
                raw_data[h].append(row.get(h, ""))

    result_arrays = {}

    for full_col_name in headers:
        parts = full_col_name.split(";")
        col_spec = parts[0]

        if "." not in col_spec:
            raise ValueError(f"列名 '{col_spec}' 缺少协议类型后缀 (例如 .PT, .I32)")

        base_name, proto_key = col_spec.rsplit(".", 1)
        proto_cls = PROTO_NAMES.get(proto_key.upper())

        if proto_cls is None:
            raise ValueError(f"未知的协议类型: {proto_key} (列: {full_col_name})")

        raw_values = raw_data[full_col_name]
        parsed_values = []

        # 根据不同的协议类型执行数据转换
        if proto_cls is Pinyin:
            # PY: 使用 pinyinparser 解析，并转为 list[list[int]]
            for val in raw_values:
                if val.strip():
                    syls = pinyin_parse(val)
                    # Syllable 实现了 __index__，可以直接转为整型
                    parsed_values.append([int(s) for s in syls])
                else:
                    parsed_values.append([])
            array_data = Pinyin.from_python(parsed_values)

        elif proto_cls is PlainText:
            # PT: 直接传入字符串列表
            array_data = PlainText.from_python(raw_values)

        elif proto_cls is Bool:
            # B: from_python 期望 list[int]，处理布尔转换
            for val in raw_values:
                val_lower = val.strip().lower()
                if val_lower in ("true", "1", "t", "yes", "y"):
                    parsed_values.append(1)
                elif val_lower in ("false", "0", "f", "no", "n", ""):
                    parsed_values.append(0)
                else:
                    raise ValueError(f"无法将 '{val}' 解析为 Bool (列: {full_col_name})")
            array_data = Bool.from_python(parsed_values)

        elif issubclass(proto_cls, _Enum):
            # EN8/16/32/64: from_python 期望 list[str]
            array_data = proto_cls.from_python(raw_values)

        elif issubclass(proto_cls, _Int):
            # I8/16/32/64, U8/16/32/64: from_python 期望 list[int]
            for val in raw_values:
                if val.strip() == "":
                    parsed_values.append(0)
                else:
                    parsed_values.append(int(val))
            array_data = proto_cls.from_python(parsed_values)

        elif issubclass(proto_cls, _Float):
            # F32/F64: from_python 期望 list[float]
            for val in raw_values:
                if val.strip() == "":
                    parsed_values.append(0.0)
                else:
                    parsed_values.append(float(val))
            array_data = proto_cls.from_python(parsed_values)

        elif issubclass(proto_cls, _Complex):
            # C64/C128: from_python 期望 list[complex]
            for val in raw_values:
                if val.strip() == "":
                    parsed_values.append(0j)
                else:
                    parsed_values.append(complex(val))
            array_data = proto_cls.from_python(parsed_values)

        else:
            raise ValueError(f"不支持的协议类型: {proto_cls}")

        # 将完整列名（含JIT部分）作为 Array 的字段名
        result_arrays[full_col_name] = array_data

    return ak.Array(result_arrays)


if sys.argv[1]:
    fn = Path(sys.argv[1])
    delim = "," if fn.suffix == "csv" else "\t"
    arr = compile_csv_tsv(fn, delim)
    ak.to_parquet(arr, fn.with_suffix(".pq"))
