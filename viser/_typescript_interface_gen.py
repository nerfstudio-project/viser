from typing import ClassVar, List, Type, get_args, get_origin, get_type_hints

import numpy as onp
from typing_extensions import Literal

from ._messages import Message

_raw_type_mapping = {
    bool: "boolean",
    float: "number",
    int: "number",
    str: "string",
    onp.ndarray: "ArrayBuffer",
}


def _get_ts_type(typ: Type) -> str:
    if get_origin(typ) is tuple:
        return "[" + ", ".join(map(_get_ts_type, get_args(typ))) + "]"
    if get_origin(typ) is Literal:
        return " | ".join(
            map(
                lambda lit: repr(lit).lower() if type(lit) is bool else repr(lit),
                get_args(typ),
            )
        )

    if hasattr(typ, "__origin__"):
        typ = typ.__origin__
    if typ in _raw_type_mapping:
        return _raw_type_mapping[typ]

    assert False, f"Unsupported type: {typ}"


def generate_typescript_defs() -> str:
    out_lines = [
        (
            "// AUTOMATICALLY GENERATED message interfaces, from Python dataclass"
            " definitions."
        ),
        "// This file should not be manually modified.",
        "",
        "// For numpy arrays, we directly serialize the underlying data buffer.",
        "type ArrayBuffer = Uint8Array;",
        "",
    ]

    def get_subclasses(typ: Type) -> List[Type[Message]]:
        out = []
        for sub in typ.__subclasses__():
            out.append(sub)
            out.extend(get_subclasses(sub))
        return out

    message_types = get_subclasses(Message)

    # Generate interfaces for each specific message.
    for cls in message_types:
        out_lines.append(f"export interface {cls.__name__} {{")
        for name, typ in get_type_hints(cls).items():
            if typ == ClassVar[str]:
                typ = f'"{getattr(cls, name)}"'
            else:
                typ = _get_ts_type(typ)
            out_lines.append(f"  {name}: {typ};")
        out_lines.append("}")
    out_lines.append("")

    # Generate union type over all messages.
    out_lines.append("export type Message = ")
    for cls in message_types:
        out_lines.append(f"  | {cls.__name__}")
    out_lines[-1] = out_lines[-1] + ";"

    return "\n".join(out_lines) + "\n"
