from typing import Optional


class Test:
    @property
    def ok(self) -> bool:
        return True
    
    @ok.setter
    def ok(self, value: Optional[bool]) -> None:
        pass


a = Test()
reveal_type(a.ok)
a.ok = False