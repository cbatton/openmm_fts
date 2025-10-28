"""Performs a natural sort on a list of strings."""

import re


def natural_sort(items: list[str]) -> list[str]:
    """Performing a natural sort on a list of strings."""

    def convert(text: str) -> int | str:
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key: str) -> list[int | str]:
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(items, key=alphanum_key)
