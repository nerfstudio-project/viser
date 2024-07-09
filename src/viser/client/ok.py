from typing import TypedDict


class SomeDict(TypedDict):
    a: int


x: SomeDict = {"a": 1}


def main(y: SomeDict) -> None: ...


main({"a": 1})
