#!/usr/bin/env python3
"""
Clean TypeScript to Python TypedDict generator.

Simple, focused approach based on lessons learned:
1. Discover interfaces and type aliases
2. Resolve types with semantic aliases for unsupported patterns
3. Generate clean Python code

Focus on high-value improvements:
- Proper interface references (cursor: Cursor not cursor: Any)
- Basic type improvements (number -> float, boolean -> bool)
- Enum literals (1 | 2 -> Literal[1, 2])
- Semantic aliases for unsupported types (functions -> JSCallback)
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
# import tyro.


@dataclass
class Interface:
    name: str
    properties: List[Dict[str, str]]
    raw_definition: str


@dataclass
class TypeAlias:
    name: str
    definition: str


@dataclass
class Enum:
    name: str
    members: List[tuple[str, str]]  # List of (member_name, value) pairs.


class TypeConverter:
    """Convert TypeScript types to Python types with semantic aliases."""

    def __init__(
        self,
        interfaces: Set[str],
        enums: Optional[Dict[str, Enum]] = None,
        type_aliases: Optional[Dict[str, str]] = None,
    ):
        self.known_interfaces = interfaces
        self.enums = enums or {}
        self.type_aliases = type_aliases or {}
        self.resolving = set()  # For cycle detection.
        self.recursion_depth = 0  # Track recursion depth.

    def convert(self, ts_type: str) -> str:
        """Convert TypeScript type to Python type with scope-aware resolution."""
        original_type = ts_type
        ts_type = (
            ts_type.strip().rstrip(";").rstrip(",")
        )  # Clean up trailing punctuation.

        # Prevent infinite recursion for any type, not just aliases.
        if ts_type in self.resolving:
            return "ComplexType"

        self.resolving.add(ts_type)
        try:
            # PRIORITY 1: Known interfaces (interfaces take precedence over type aliases).
            if ts_type in self.known_interfaces:
                return ts_type

            # PRIORITY 2: Exact type alias matches.
            if ts_type in self.type_aliases:
                resolved = self.convert(self.type_aliases[ts_type])
                return resolved

            # PRIORITY 3: Namespaced type resolution.
            if "." in ts_type:
                parts = ts_type.split(".")
                unqualified = parts[
                    -1
                ]  # Get the last part (e.g., "Rotate" from "Axis.Rotate").
                if unqualified in self.type_aliases:
                    resolved = self.convert(self.type_aliases[unqualified])
                    return resolved

            return self._convert_basic(ts_type)
        finally:
            self.resolving.discard(ts_type)

    def _convert_basic(self, ts_type: str) -> str:
        """Convert basic TypeScript type without recursion protection."""

        # Basic types.
        basic_types = {
            "string": "str",
            "number": "float",
            "boolean": "bool",
            "void": "None",
            "null": "None",
            "undefined": "None",
            "any": "Any",
            "unknown": "Any",
        }

        if ts_type in basic_types:
            return basic_types[ts_type]

        # String/number literals.
        if (
            (ts_type.startswith('"') and ts_type.endswith('"'))
            or (ts_type.startswith("'") and ts_type.endswith("'"))
            or ts_type.replace(".", "").replace("-", "").isdigit()
        ):
            return f"Literal[{ts_type}]"

        # Check known enums first.
        if ts_type in self.enums:
            # Return the enum class name directly.
            return ts_type

        # Try namespaced enum lookup (e.g., Axis.Align -> Align).
        if "." in ts_type:
            parts = ts_type.split(".")
            unqualified = parts[-1]
            if unqualified in self.enums:
                # Return the enum class name directly.
                return unqualified

        # Arrays - handle parenthesized unions first.
        if ts_type.endswith("[]"):
            inner = ts_type[:-2]
            # Handle parenthesized unions like (number | null).
            if inner.startswith("(") and inner.endswith(")"):
                inner = inner[1:-1]  # Remove outer parentheses.
            inner_converted = self.convert(inner)
            return f"list[{inner_converted}]"

        # Tuples [min: number, max: number].
        tuple_match = re.match(r"^\[([^\]]+)\]$", ts_type)
        if tuple_match:
            content = tuple_match.group(1)
            parts = [p.strip() for p in content.split(",")]
            elements = []

            for part in parts:
                if ":" in part:
                    # Remove label: "min: number" -> "number".
                    type_part = part.split(":", 1)[1].strip()
                    elements.append(self.convert(type_part))
                else:
                    elements.append(self.convert(part))

            return f"tuple[{', '.join(elements)}]"

        # Function types -> JSCallback (process BEFORE union types to avoid confusion with | in params).
        # Match: (params) => return or Arrow function syntax, but not unions containing functions.
        if ("=>" in ts_type and (ts_type.startswith("(") or not "|" in ts_type)) or (
            ts_type.startswith("(")
            and ")" in ts_type
            and not ts_type.endswith("[]")
            and not "|" in ts_type
        ):
            return "JSCallback"

        # Union types (process AFTER function types).
        if "|" in ts_type:
            # Smart split that respects parentheses.
            parts = self._split_union(ts_type)
            converted = []

            for part in parts:
                converted_part = self.convert(part)
                converted.append(converted_part)

            # Remove duplicates while preserving order.
            seen = set()
            unique_converted = []
            for item in converted:
                if item not in seen:
                    seen.add(item)
                    unique_converted.append(item)

            if len(unique_converted) > 1:
                return f"{' | '.join(unique_converted)}"
            elif len(unique_converted) == 1:
                return unique_converted[0]
            else:
                return "ComplexType"

        # DOM/Browser types.
        dom_types = ["HTMLElement", "Element", "Node", "Event", "MouseEvent", "DOMRect"]
        if any(dom in ts_type for dom in dom_types):
            return "DOMElement"

        # Canvas/CSS properties.
        if "CanvasRenderingContext2D" in ts_type or "CSSStyleDeclaration" in ts_type:
            return "CSSValue"

        # Try namespaced interface lookup (e.g., Axis.Grid -> Axis_Grid).
        if "." in ts_type:
            parts = ts_type.split(".")
            underscore_name = "_".join(parts)
            if underscore_name in self.known_interfaces:
                return underscore_name

        # Type aliases should have been resolved above, but check for unknown qualified names.
        if "." in ts_type and re.match(r"^[A-Z][a-zA-Z0-9_.]*$", ts_type):
            return "UnknownType"

        # PascalCase types that might be interfaces we missed.
        if re.match(r"^[A-Z][a-zA-Z0-9_]*$", ts_type):
            return "UnknownType"

        # Everything else.
        return "ComplexType"

    def _split_union(self, ts_type: str) -> List[str]:
        """Split union type on | while respecting parentheses and brackets."""
        parts = []
        current_part = ""
        paren_depth = 0
        bracket_depth = 0

        for char in ts_type:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            elif char == "|" and paren_depth == 0 and bracket_depth == 0:
                # This is a top-level union separator.
                parts.append(current_part.strip())
                current_part = ""
                continue

            current_part += char

        # Add the last part.
        if current_part.strip():
            parts.append(current_part.strip())

        return parts


def parse_type_aliases(content: str) -> Dict[str, str]:
    """Parse type alias declarations from TypeScript content with proper namespace tracking."""
    type_aliases = {}
    lines = content.split("\n")
    i = 0
    namespace_stack = []
    namespace_brace_depth = []  # Track brace depth for each namespace.
    current_brace_depth = 0

    while i < len(lines):
        line = lines[i].strip()

        # Count braces to track nesting depth..
        open_braces = line.count("{")
        close_braces = line.count("}")

        # Track namespace entries.
        namespace_match = re.match(r"(export |declare )?namespace (\w+)", line)
        if namespace_match:
            namespace_name = namespace_match.group(2)
            namespace_stack.append(namespace_name)
            # After this line, we'll be one level deeper.
            namespace_brace_depth.append(current_brace_depth + open_braces)

        # Update current brace depth.
        current_brace_depth += open_braces - close_braces

        # Check if we've exited any namespaces.
        while namespace_stack and current_brace_depth < namespace_brace_depth[-1]:
            namespace_stack.pop()
            namespace_brace_depth.pop()

        # Find type alias declarations: export type Show = boolean | ...
        type_match = re.match(r"export type (\w+)\s*=\s*(.+)", line)
        if type_match:
            type_name = type_match.group(1)
            type_def = type_match.group(2).rstrip(";")

            # Build qualified name with namespace (skip main uPlot namespace).
            relevant_namespaces = [ns for ns in namespace_stack if ns != "uPlot"]
            if relevant_namespaces:
                qualified_name = ".".join(relevant_namespaces + [type_name])
                type_aliases[qualified_name] = type_def
                # Also add unqualified name for fallback.
                type_aliases[type_name] = type_def
            else:
                type_aliases[type_name] = type_def

        i += 1

    return type_aliases


def parse_enums(content: str) -> Dict[str, Enum]:
    """Parse const enum declarations from TypeScript content."""
    enums = {}
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Find const enum declarations.
        enum_match = re.match(r"export const enum (\w+)", line)
        if enum_match:
            enum_name = enum_match.group(1)

            # Extract enum body.
            enum_members = []
            i += 1

            while i < len(lines):
                current_line = lines[i].strip()

                if current_line == "}":
                    break

                # Parse enum member: Name = value,.
                member_match = re.match(r"(\w+)\s*=\s*(-?\d+)", current_line)
                if member_match:
                    member_name = member_match.group(1)
                    value = member_match.group(2)
                    enum_members.append((member_name, value))

                i += 1

            if enum_members:
                enums[enum_name] = Enum(name=enum_name, members=enum_members)

        i += 1

    return enums


def parse_interfaces(content: str) -> List[Interface]:
    """Parse interfaces from TypeScript content with namespace support."""
    interfaces = []
    lines = content.split("\n")
    i = 0
    namespace_stack = []

    while i < len(lines):
        line = lines[i].strip()

        # Track namespace entries.
        namespace_match = re.match(r"(export |declare )?namespace (\w+)", line)
        if namespace_match:
            namespace_name = namespace_match.group(2)
            namespace_stack.append(namespace_name)
            i += 1
            continue

        # Track namespace exits (simple heuristic).
        if line == "}" and namespace_stack:
            indent = len(lines[i]) - len(lines[i].lstrip())
            # If we're at low indentation, probably exiting a namespace.
            if indent <= len(namespace_stack):
                namespace_stack.pop()

        # Find interface declarations.
        interface_match = re.match(r"export interface (\w+)", line)
        if interface_match:
            interface_name = interface_match.group(1)

            # Build qualified name with namespace (skip main uPlot namespace for top-level exports).
            if namespace_stack and not (
                len(namespace_stack) == 1 and namespace_stack[0] == "uPlot"
            ):
                qualified_name = "_".join(namespace_stack + [interface_name])
            else:
                qualified_name = interface_name

            # Extract interface body.
            body_lines = []
            brace_count = 0
            started = False

            while i < len(lines):
                current_line = lines[i]

                if "{" in current_line:
                    started = True

                if started:
                    body_lines.append(current_line)
                    brace_count += current_line.count("{") - current_line.count("}")
                    if brace_count == 0:
                        break

                i += 1

            # Parse properties.
            properties = parse_properties("\n".join(body_lines))

            interfaces.append(
                Interface(
                    name=qualified_name,
                    properties=properties,
                    raw_definition="\n".join(body_lines),
                )
            )

        i += 1

    return interfaces


def parse_properties(interface_body: str) -> List[Dict[str, str]]:
    """Parse properties from interface body."""
    properties = []
    lines = interface_body.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line in ["{", "}"]:
            i += 1
            continue

        # Collect comments.
        comment = ""
        while (
            line.startswith("/**")
            or line.startswith("//")
            or (line.startswith("*") and ":" not in line)
        ):
            if line.startswith("/**"):
                comment = line[3:].strip()
            elif line.startswith("//"):
                comment = line[2:].strip()
            elif line.startswith("*"):
                comment = line[1:].strip()

            if comment.endswith("*/"):
                comment = comment[:-2].strip()

            i += 1
            line = lines[i].strip() if i < len(lines) else ""

        # Parse index signatures like [key: string]: Scale.
        if line.startswith("[") and "]:" in line:
            # Extract index signature: [key: string]: Scale -> key_type: string, value_type: Scale.
            match = re.match(r"^\[([^:]+):\s*([^\]]+)\]:\s*(.+);?$", line.strip())
            if match:
                key_name = match.group(1).strip()
                key_type = match.group(2).strip()
                value_type = match.group(3).strip().rstrip(";").rstrip(",")

                # Create a special property to indicate this is an index signature.
                properties.append(
                    {
                        "name": "__index_signature__",
                        "type": f"[{key_type}, {value_type}]",  # Store both key and value types.
                        "optional": False,
                        "comment": f"Index signature: {key_name} -> {value_type}",
                    }
                )

        # Parse regular property line.
        elif ":" in line and not line.startswith("["):
            # Handle multi-line properties - but be careful about parsing.
            prop_line = line

            # Only continue to next line if this line doesn't end with ; or ,.
            while (
                not prop_line.rstrip().endswith(";")
                and not prop_line.rstrip().endswith(",")
                and i + 1 < len(lines)
            ):
                i += 1
                next_line = lines[i].strip()
                # Stop if next line looks like a new property or comment.
                if (
                    ":" in next_line
                    or next_line.startswith("/**")
                    or next_line.startswith("//")
                ):
                    i -= 1  # Back up.
                    break
                prop_line += " " + next_line

            prop = parse_property(prop_line, comment)
            if prop:
                properties.append(prop)

        i += 1

    return properties


def parse_property(line: str, comment: str = "") -> Optional[Dict[str, str | bool]]:
    """Parse a single property line."""
    line = line.strip().rstrip(";").rstrip(",")

    if ":" not in line:
        return None

    optional = "?" in line
    parts = line.split(":", 1)
    if len(parts) != 2:
        return None

    name = parts[0].replace("?", "").strip()
    type_str = parts[1].strip()

    # Clean up type string - remove everything after comma or comment.
    if "," in type_str:
        type_str = type_str.split(",")[0].strip()

    # Remove inline comments.
    if "//" in type_str:
        type_str = type_str.split("//")[0].strip()

    if "/*" in type_str:
        type_str = type_str.split("/*")[0].strip()

    # Handle readonly.
    if name.startswith("readonly "):
        name = name[9:]

    # Validate property name.
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        return None

    return {"name": name, "type": type_str, "optional": optional, "comment": comment}


def escape_python_keyword(name: str) -> str:
    """Escape Python keywords by appending underscore."""
    python_keywords = {
        "and",
        "as",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "exec",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "not",
        "or",
        "pass",
        "print",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        "None",
        "True",
        "False",
    }
    return name + "_" if name in python_keywords else name


def camel_to_screaming_snake(name: str) -> str:
    """Convert camelCase to SCREAMING_SNAKE_CASE for enum members."""
    # Insert underscores before capital letters (except the first)
    result = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return result.upper()


def topological_sort_interfaces(interfaces: List[Interface]) -> List[Interface]:
    """Sort interfaces in topological order so dependencies come before dependents."""
    from collections import defaultdict, deque

    interface_map = {iface.name: iface for iface in interfaces}
    interface_names = set(interface_map.keys())

    # Build dependency graph: interface -> set of interfaces it depends on.
    dependencies = defaultdict(set)
    dependents = defaultdict(set)  # reverse mapping

    for interface in interfaces:
        interface_name = interface.name
        deps = set()

        # Find all interface dependencies in properties.
        for prop in interface.properties:
            # Skip index signatures.
            if prop["name"] == "__index_signature__":
                continue

            prop_type = prop["type"]

            # Find interface references in the type string.
            # Look for exact interface names (including underscore names like Legend_Markers).
            for other_name in interface_names:
                if other_name != interface_name:
                    # Use word boundary regex to ensure exact matches.
                    # But exclude patterns like "Legend.Something" which are type aliases, not interface refs.
                    if re.search(r"\b" + re.escape(other_name) + r"\b", prop_type):
                        # Make sure it's not a qualified name like "Legend.Width".
                        # by checking if it's followed by a dot and identifier.
                        full_match = re.search(
                            r"\b"
                            + re.escape(other_name)
                            + r"(\.[A-Za-z_][A-Za-z0-9_]*)?",
                            prop_type,
                        )
                        if full_match and full_match.group(1):
                            # This is a qualified name like "Legend.Width", not an interface reference.
                            continue
                        deps.add(other_name)

            # Also look for namespaced references that map to underscore names.
            # e.g., "Legend.Markers" should create dependency on "Legend_Markers".
            namespaced_refs = re.findall(
                r"\b[A-Z][a-zA-Z0-9_]*\.[A-Z][a-zA-Z0-9_]*\b", prop_type
            )
            for ref in namespaced_refs:
                underscore_name = ref.replace(".", "_")
                if (
                    underscore_name in interface_names
                    and underscore_name != interface_name
                ):
                    deps.add(underscore_name)

        dependencies[interface_name] = deps
        for dep in deps:
            dependents[dep].add(interface_name)

    # Debug output.
    for name, deps in dependencies.items():
        if deps:
            print(f"Debug: {name} depends on: {deps}")

    # Kahn's algorithm for topological sorting.
    # Start with interfaces that have no dependencies.
    in_degree = {iface.name: len(dependencies[iface.name]) for iface in interfaces}
    queue = deque([name for name, degree in in_degree.items() if degree == 0])
    result = []

    while queue:
        current = queue.popleft()
        result.append(interface_map[current])

        # Remove this interface from its dependents' dependency lists.
        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Check for cycles.
    if len(result) != len(interfaces):
        remaining = set(interface_map.keys()) - {iface.name for iface in result}
        print(f"Warning: Circular dependencies detected in interfaces: {remaining}")
        # Add remaining interfaces in original order.
        for interface in interfaces:
            if interface.name in remaining:
                result.append(interface)

    return result


def find_needed_interfaces(
    interfaces: List[Interface], start: str = "Options"
) -> Set[str]:
    """Find interfaces needed starting from the given interface."""
    needed = set()
    to_process = [start]
    interface_map = {iface.name: iface for iface in interfaces}

    while to_process:
        current = to_process.pop()
        if current in needed or current not in interface_map:
            continue

        needed.add(current)

        # Look for interface references in properties.
        for prop in interface_map[current].properties:
            # Find PascalCase words that might be interface names.
            words = re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", prop["type"])
            for word in words:
                if word in interface_map and word not in needed:
                    to_process.append(word)

            # Also look for namespaced interfaces like Axis.Grid -> Axis_Grid.
            namespaced_refs = re.findall(
                r"\b[A-Z][a-zA-Z0-9_]*\.[A-Z][a-zA-Z0-9_]*\b", prop["type"]
            )
            for ref in namespaced_refs:
                underscore_name = ref.replace(".", "_")
                if underscore_name in interface_map and underscore_name not in needed:
                    to_process.append(underscore_name)

    return needed


def generate_python_code(interfaces: List[Interface], content: str) -> str:
    """Generate clean Python TypedDict code."""

    # Find needed interfaces.
    needed_names = find_needed_interfaces(interfaces, "Options")
    needed_interfaces = [iface for iface in interfaces if iface.name in needed_names]

    # Sort interfaces in topological order to eliminate forward references.
    needed_interfaces = topological_sort_interfaces(needed_interfaces)

    # Parse enums and type aliases from content.
    enums = parse_enums(content)
    type_aliases = parse_type_aliases(content)

    # Create type converter.
    # Include enum names as known types along with interface names.
    known_types = {iface.name for iface in needed_interfaces} | set(enums.keys())
    converter = TypeConverter(known_types, enums, type_aliases)

    lines = [
        '"""Clean uPlot TypedDict definitions."""',
        "",
        "from __future__ import annotations",
        "",
        "from enum import IntEnum",
        "from typing import Any, Dict, Literal",
        "from typing_extensions import Never, Required, TypedDict",
        "",
        "# Semantic type aliases for unsupported/complex TypeScript patterns",
        "JSCallback = Never      # JavaScript function signatures",
        "DOMElement = Never      # DOM elements (HTMLElement, etc.)",
        "CSSValue = str          # CSS property values",
        "UnknownType = Any       # Unknown interface references",
        "ComplexType = Any       # Complex TypeScript patterns",
        "",
    ]

    # Generate IntEnum classes.
    if enums:
        lines.append("# Enum definitions")
        for enum_name, enum_obj in enums.items():
            lines.append(f"class {enum_name}(IntEnum):")
            for member_name, value in enum_obj.members:
                snake_case_name = camel_to_screaming_snake(member_name)
                lines.append(f"    {snake_case_name} = {value}")
            lines.append("")

    # Generate TypedDict classes and collect type aliases.
    type_aliases_to_generate = []
    interface_names = {iface.name for iface in needed_interfaces}

    # First pass: collect type aliases that need to be generated.
    for interface in needed_interfaces:
        name = interface.name
        properties = interface.properties

        # Check for index signatures
        index_signatures = [p for p in properties if p["name"] == "__index_signature__"]
        regular_properties = [
            p for p in properties if p["name"] != "__index_signature__"
        ]

        # Handle interfaces with only index signatures (like Scales).
        if index_signatures and not regular_properties:
            # Store for generation before classes.
            index_sig = index_signatures[0]
            # Parse [key_type, value_type] from the stored format.
            match = re.match(r"^\[([^,]+),\s*(.+)\]$", index_sig["type"])
            if match:
                key_type = converter.convert(match.group(1).strip())
                value_type = converter.convert(match.group(2).strip())
                type_aliases_to_generate.append(
                    (name, index_sig["comment"], key_type, value_type)
                )

    # Generate type aliases first (using string quotes for forward references).
    if type_aliases_to_generate:
        lines.append("# Type aliases for index signatures")
        for name, comment, key_type, value_type in type_aliases_to_generate:
            # Use string quotes to allow forward references.
            value_type_str = (
                f'"{value_type}"' if value_type in interface_names else value_type
            )
            # Add period to comment if it doesn't already end with punctuation.
            if comment and not comment.rstrip().endswith((".", "!", "?", ":", ";")):
                comment += "."
            lines.extend(
                [f"# {comment}", f"{name} = Dict[{key_type}, {value_type_str}]", ""]
            )

    # Second pass: generate TypedDict classes.
    type_alias_names = {name for name, _, _, _ in type_aliases_to_generate}

    for interface in needed_interfaces:
        name = interface.name

        # Skip if this was already handled as a type alias.
        if name in type_alias_names:
            continue

        properties = interface.properties

        # Check for index signatures
        index_signatures = [p for p in properties if p["name"] == "__index_signature__"]
        regular_properties = [
            p for p in properties if p["name"] != "__index_signature__"
        ]

        if not regular_properties:
            lines.extend([f"{name} = TypedDict('{name}', {{}}, total=False)", ""])
            continue

        # Separate required and optional regular properties
        required = [p for p in regular_properties if not p["optional"]]
        optional = [p for p in regular_properties if p["optional"]]

        # Generate functional TypedDict
        lines.append(f"{name} = TypedDict('{name}', {{")

        # Add all properties (both required and optional)
        all_props = required + optional
        for i, prop in enumerate(all_props):
            prop_name = prop["name"]  # Use original name, no escaping needed in quotes
            prop_type = converter.convert(prop["type"])

            # Add comment above the property
            if prop["comment"]:
                comment = prop["comment"]
                # Add period if comment doesn't already end with punctuation
                if comment and not comment.rstrip().endswith((".", "!", "?", ":", ";")):
                    comment += "."
                lines.append(f"    # {comment}")

            # Handle required vs optional
            if required and optional:  # Mixed case - use Required[] for required props
                if prop in required:
                    type_annotation = f"Required[{prop_type}]"
                else:
                    type_annotation = prop_type
            else:
                # All required or all optional
                type_annotation = prop_type

            # Add trailing comma except for last item
            comma = "," if i < len(all_props) - 1 else ""
            lines.append(f"    '{prop_name}': {type_annotation}{comma}")

        # Set total=False if we have any optional properties
        total_param = ", total=False" if optional else ""
        lines.append(f"}}{total_param})")
        lines.append("")

    return "\n".join(lines)


def main(
    input_file: Path = Path("uPlot.d.ts"),
    output_file: Path = Path("uplot_types_generated.py"),
) -> None:
    """Generate clean Python TypedDict types from TypeScript definitions."""

    print(f"Reading {input_file}")
    content = input_file.read_text()

    print("Parsing interfaces...")
    interfaces = parse_interfaces(content)
    print(f"Found {len(interfaces)} interfaces")

    print("Generating Python code...")
    python_code = generate_python_code(interfaces, content)

    print(f"Writing to {output_file}")
    output_file.write_text(python_code)

    print("Done!")


if __name__ == "__main__":
    main()
