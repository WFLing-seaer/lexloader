import csv
import typing

import awkward as ak

from colproto import PROTO_NAMES, ColProtoABC

# 1. 读取 TSV
path_in = r"Z:\乡以上行政区划.tsv"
path_out = r"Z:\乡以上行政区划.pq"

with open(path_in, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    headers = reader.fieldnames
    rows = list(reader)

# 2. 按列拆分，解析列名与协议后缀
col_data = {}
for hdr in headers:
    proto_tag = hdr.rsplit(".", 1)[-1]
    proto_cls = PROTO_NAMES[proto_tag]

    # 通过 __mro__ 回溯找到 ColProtoABC[T] 中的 T，作为原始值类型
    typ = None
    for base_cls in proto_cls.__mro__:
        for base in getattr(base_cls, "__orig_bases__", []):
            if typing.get_origin(base) is ColProtoABC:
                args = typing.get_args(base)
                if args:
                    typ = args[0]
                    break
        if typ is not None:
            break

    # 用 T 将 TSV 中读出的 str 统一转换为正确类型
    raw = [typ(row[hdr]) for row in rows]
    col_data[hdr] = proto_cls.from_python(raw)

# 3. 用各列的 .data 构建 Awkward Array
fields = {col_name: inst.data for col_name, inst in col_data.items()}
awk_arr = ak.Array(fields)

# 4. 保存为 Parquet
ak.to_parquet(awk_arr, path_out)
