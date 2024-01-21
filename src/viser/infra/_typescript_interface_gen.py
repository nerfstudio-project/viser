from functools import partial
import dataclasses
from typing import Any, ClassVar, Type, Union, cast, get_type_hints, Dict, Tuple

import numpy as onp
from typing_extensions import Literal, get_args, get_origin, is_typeddict

try:
    from typing import Literal as LiteralAlt
except ImportError:
    LiteralAlt = Literal  # type: ignore

from ._messages import Message
from .._gui_components import Component, Property

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


# def _is_property(typ: Type[Any]) -> bool:
#     if get_origin(typ) is not Union:
#         return False
#     if len(get_args(typ)) != 2:
#         return False
#     base_type = get_args(typ)[0]
#     link_type = get_args(type)[1]
#     if get_origin(link_type) is not Link:
#     get_args(typ) == (Link,)


def _get_ts_type(typ: Type[Any], known_types: Dict[Type, str]) -> str:
    origin_typ = get_origin(typ)

    if typ in known_types:
        return known_types[typ]
    if origin_typ is tuple:
        args = get_args(typ)
        if len(args) == 2 and args[1] == ...:
            return _get_ts_type(args[0], known_types) + "[]"
        else:
            return "[" + ", ".join(map(partial(_get_ts_type, known_types=known_types), args)) + "]"
    elif origin_typ in (Literal, LiteralAlt):
        return " | ".join(
            map(
                lambda lit: repr(lit).lower() if type(lit) is bool else repr(lit),
                get_args(typ),
            )
        )
    elif origin_typ is Property:
        subtype = _get_ts_type(get_args(typ)[0], known_types)
        return "Property<" + subtype + ">"
    elif origin_typ is Union:
        return (
            "("
            + " | ".join(
                map(
                    partial(_get_ts_type, known_types=known_types),
                    get_args(typ),
                )
            )
            + ")"
        )
    elif is_typeddict(typ):
        hints = get_type_hints(typ)

        def fmt(key):
            val = hints[key]
            ret = f"'{key}'" + ": " + _get_ts_type(val, known_types)
            return ret

        ret = "{" + ", ".join(map(fmt, hints)) + "}"
        return ret
    else:
        # Like get_origin(), but also supports numpy.typing.NDArray[dtype].
        typ = cast(Any, getattr(typ, "__origin__", typ))

        if typ not in _raw_type_mapping:
            breakpoint()

        assert typ in _raw_type_mapping, f"Unsupported type {typ}"
        return _raw_type_mapping[typ]


def generate_typescript_components() -> str:
    component_types = get_args(Component)
    known_types = {t: t.__name__ + "Props" for t in component_types}
    known_types[Component] = "AllComponentProps"

    lines = []
    lines.append("import {")
    lines.append("  AllComponentProps,")
    for cls in component_types:
        lines.append(f"  {cls.__name__ + 'Props'},")
    lines.append("} from './WebsocketMessages';")
    for cls in component_types:
        cname = cls.__name__
        lines.append(f"import {cname} from './components/{cname}';")
    lines.append("")
    lines.append("")
    lines.append("export default function GeneratedComponent({type, props}: AllComponentProps) {")
    lines.append("  switch (type) {")
    for cls in component_types:
        lines.append(f'    case "{cls.__name__}":')
        lines.append(f"      return <{cls.__name__} {{...props}} />;")
    
    lines.append("    default:")
    lines.append("      throw new Error(`Unknown component type ${type}`);")
    lines.append("  }")
    lines.append("}")

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
        + "\n".join(lines) + "\n"
    )


def generate_typescript_interfaces(message_cls: Type[Message]) -> Tuple[str, str]:
    """Generate TypeScript definitions for all subclasses of a base message class."""
    out_lines = []
    message_types = message_cls.get_subclasses()
    component_types = get_args(Component)
    known_types = {t: t.__name__ for t in component_types}
    known_types[Component] = "AllComponentProps"

    # Add common property interface.
    out_lines.append("export type Property<T> = { path: string; } | { value: T };")
    out_lines.append("")

    # Generate interfaces for each specific message.
    for types, postfix in [(component_types, "Props"), (message_types, "")]:
        for cls in types:
            if cls.__doc__ is not None:
                docstring = "\n * ".join(
                    map(lambda line: line.strip(), cls.__doc__.split("\n"))
                )
                out_lines.append(f"/** {docstring}")
                out_lines.append(" *")
                out_lines.append(" * (automatically generated)")
                out_lines.append(" */")

            out_lines.append(f"export interface {cls.__name__ + postfix} " + "{")
            out_lines.append(f'  type: "{cls.__name__}";')
            field_names = set([f.name for f in dataclasses.fields(cls)])  # type: ignore
            for name, typ in get_type_hints(cls).items():
                if typ == ClassVar[str]:
                    typ = f'"{getattr(cls, name)}"'
                elif name in field_names:
                    typ = _get_ts_type(typ, known_types)
                else:
                    continue
                out_lines.append(f"  {name}: {typ};")
            out_lines.append("}")
        out_lines.append("")

    # Generate union type over all component props.
    out_lines.append("export type AllComponentProps = ")
    for cls in component_types:
        out_lines.append(f"  | {cls.__name__ + 'Props'}")
    out_lines[-1] = out_lines[-1] + ";"

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
