from pathlib import Path
import collections.abc
from typing import get_origin, get_args
import typing
import inspect
from viser import _gui_api_core as gui_api
from viser import _gui_handles as gui_handles


class StubGenerator:
    def __init__(self):
        self._text = "import typing\n\n"
        self._indent = " " * 4

    def _get_annotation(self, annotation):
        origin = get_origin(annotation)
        if origin is typing.Union:
            if type(None) in get_args(annotation):
                args = [arg for arg in get_args(annotation) if arg is not type(None)]
                return "typing.Optional[" + self._get_annotation(args[0]) + "]"
            args = get_args(annotation)
            return (
                "typing.Union["
                + ", ".join(self._get_annotation(arg) for arg in args)
                + "]"
            )
        if origin is typing.Callable or origin is collections.abc.Callable:
            args = get_args(annotation)
            return (
                "typing.Callable[["
                + ", ".join(self._get_annotation(arg) for arg in args[0])
                + "], "
                + self._get_annotation(args[1])
                + "]"
            )
        if isinstance(annotation, typing.ForwardRef):
            return f"'{annotation.__forward_arg__}'"
        if annotation is inspect.Parameter.empty:
            return "typing.Any"
        if annotation is type(None):
            return "None"
        if inspect.isclass(origin):
            args = get_args(annotation)
            return (
                origin.__name__
                + "["
                + ", ".join(self._get_annotation(arg) for arg in args)
                + "]"
            )
        if inspect.isclass(annotation):
            return annotation.__name__
        return str(annotation)

    def add_function(self, method, offset=0):
        self._text += self._indent * offset + f"def {method.__name__}("
        signature = inspect.signature(method)
        for name, param in signature.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                self._text += f"{name}"
                if param.default is not inspect.Parameter.empty:
                    self._text += f"={repr(param.default)}"
            else:
                self._text += f"{name}: {self._get_annotation(param.annotation)}"
                if param.default is not inspect.Parameter.empty:
                    self._text += f" = {repr(param.default)}"
            self._text += ", "
        if signature.return_annotation is inspect.Signature.empty:
            self._text = self._text[:-2] + "):\n"
        else:
            self._text = self._text[:-2] + ") -> "
            self._text += self._get_annotation(signature.return_annotation) + ":\n"
        doc = inspect.getdoc(method)
        if doc:
            self._text += self._indent * (offset + 1) + '"""' + doc + '"""\n'
        self._text += self._indent * (offset + 1) + "...\n\n"

    def add_class(self, cls):
        self._text += f"class {cls.__name__}:\n"
        if hasattr(cls, "__doc__") and cls.__doc__:
            self._text += f'    """{cls.__doc__}"""\n'
        has_content = False
        if hasattr(cls, "__annotations__"):
            for attr in cls.__annotations__:
                self._text += (
                    f"    {attr}: {self._get_annotation(cls.__annotations__[attr])}\n"
                )
                has_content = True
        for name, member in inspect.getmembers(cls):
            if inspect.isfunction(member):
                self.add_function(member, offset=1)
                has_content = True
        if not has_content:
            self._text += "    pass\n"
        self._text += "\n"

    @classmethod
    def from_module(cls, module) -> "StubGenerator":
        generator = cls()
        doc = inspect.getdoc(module)
        if doc:
            generator._text += '"""' + doc + '"""\n'
        for name, member in inspect.getmembers(module):
            member_module = getattr(member, "__module__", None)
            if member_module is not None and member_module != module.__name__:
                breakpoint()

        for name, member in inspect.getmembers(module):
            member_module = getattr(member, "__module__", None)
            if member_module is not None and member_module != module.__name__:
                continue
            if inspect.isclass(member):
                generator.add_class(member)
            if inspect.isfunction(member):
                generator.add_function(member)
        return generator

    def save(self, path):
        with open(path, "w") as f:
            f.write(self._text)


def generate_stubs():
    StubGenerator.from_module(gui_api).save(gui_api.__file__ + "i")
    StubGenerator.from_module(gui_handles).save(gui_handles.__file__ + "i")


if __name__ == "__main__":
    generate_stubs()
