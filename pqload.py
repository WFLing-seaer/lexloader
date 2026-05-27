import awkward as ak

from .colproto import JIT_NAMES, PROTO_NAMES, ColProtoABC


def load_columns(record_array: ak.Array) -> dict[str, ColProtoABC]:
    result: dict[str, ColProtoABC] = {}

    fields = record_array.fields

    for col_spec in fields:
        col_data = record_array[col_spec]

        parts = col_spec.split(";")

        base_name, proto_key = parts[0].rsplit(".", 1)
        proto_cls = PROTO_NAMES.get(proto_key.upper())
        if proto_cls is None:
            raise ValueError(f"未知的协议类型: {proto_key} (列: {col_spec})")

        proto_instance = proto_cls(data=col_data)
        result[base_name] = proto_instance

        for jit_part in parts[1:]:
            jit_name, jit_proto_key = jit_part.rsplit(".", 1)
            jit_cls = JIT_NAMES.get(jit_proto_key.upper())
            if jit_cls is None:
                raise ValueError(f"未知的 JIT 协议类型: {jit_proto_key} (列: {col_spec})")

            jit_instance = jit_cls(from_=proto_instance)
            result[jit_name] = jit_instance

    return result
