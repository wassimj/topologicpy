from __future__ import annotations


class IntAttribute:
    def __init__(self, value: int):
        self._value = int(value)

    def IntValue(self):
        return self._value


class DoubleAttribute:
    def __init__(self, value: float):
        self._value = float(value)

    def DoubleValue(self):
        return self._value


class StringAttribute:
    def __init__(self, value: str):
        self._value = str(value)

    def StringValue(self):
        return self._value


class ListAttribute:
    def __init__(self, values=None):
        self._values = list(values or [])

    def ListValue(self):
        return self._values
