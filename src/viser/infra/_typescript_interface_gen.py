import dataclasses
from typing import Any, ClassVar, Type, Union, get_type_hints

import numpy as onp
from typing_extensions import Literal, get_args, get_origin, is_typeddict

try:
    from typing import Literal as LiteralAlt
except ImportError:
    LiteralAlt = Literal  # type: ignore

from ._messages import Message

_raw_type_mapping = {
    bool: "boolean",
    float: "number",
    int: "number",
    str: "string",
    # For numpy arrays, we directly serialize the underlying data buffer.
    onp.ndarray: "Uint8Array",
    bytes: "Uint8Array",
    Any: "any",
    None: "null",
    type(None): "null",
}


def _get_ts_type(typ: Type[Any]) -> str:
    origin_typ = get_origin(typ)

    if origin_typ is tuple:
        args = get_args(typ)
        if len(args) == 2 and args[1] == ...:
            return _get_ts_type(args[0]) + "[]"
        else:
            return "[" + ", ".join(map(_get_ts_type, args)) + "]"
    elif origin_typ in (Literal, LiteralAlt):
        return " | ".join(
            map(
                lambda lit: repr(lit).lower() if type(lit) is bool else repr(lit),
                get_args(typ),
            )
        )
    elif origin_typ is Union:
        return (
            "("
            + " | ".join(
                map(
                    _get_ts_type,
                    get_args(typ),
                )
            )
            + ")"
        )
    elif is_typeddict(typ):
        hints = get_type_hints(typ)

        def fmt(key):
            val = hints[key]
            ret = f"'{key}'" + ": " + _get_ts_type(val)
            return ret

        ret = "{" + ", ".join(map(fmt, hints)) + "}"
        return ret
    else:
        # Like get_origin(), but also supports numpy.typing.NDArray[dtype].
        typ = getattr(typ, "__origin__", typ)

        assert typ in _raw_type_mapping, f"Unsupported type {typ}"
        return _raw_type_mapping[typ]


def generate_typescript_interfaces(message_cls: Type[Message]) -> str:
    """Generate TypeScript definitions for all subclasses of a base message class."""
    out_lines = []
    message_types = message_cls.get_subclasses()

    # Generate interfaces for each specific message.
    for cls in message_types:
        if cls.__doc__ is not None:
            docstring = "\n * ".join(
                map(lambda line: line.strip(), cls.__doc__.split("\n"))
            )
            out_lines.append(f"/** {docstring}")
            out_lines.append(" *")
            out_lines.append(" * (automatically generated)")
            out_lines.append(" */")

        out_lines.append(f"export interface {cls.__name__} " + "{")
        out_lines.append(f'  type: "{cls.__name__}";')
        field_names = set([f.name for f in dataclasses.fields(cls)])  # type: ignore
        for name, typ in get_type_hints(cls).items():
            if typ == ClassVar[str]:
                typ = f'"{getattr(cls, name)}"'
            elif name in field_names:
                typ = _get_ts_type(typ)
            else:
                continue
            out_lines.append(f"  {name}: {typ};")
        out_lines.append("}")
    out_lines.append("")

    # Generate union type over all messages.
    out_lines.append("export type Message = ")
    for cls in message_types:
        out_lines.append(f"  | {cls.__name__}")
    out_lines[-1] = out_lines[-1] + ";"

    interfaces = "\n".join(out_lines) + "\n"

    # Add header and return.
    return (
        "\n".join(
            [
                (
                    "// AUTOMATICALLY GENERATED message interfaces, from Python"
                    " dataclass definitions."
                ),
                "// This file should not be manually modified.",
                "",
            ]
        )
        + interfaces
    )
