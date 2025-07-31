"""Performs a natural sort on a list of strings."""

import re


def natural_sort(items):
    """Performing a natural sort on a list of strings."""

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(items, key=alphanum_key)
