"""
Test lib/calc.py
"""

from libs.calc import add


def test_add():
    assert add(1, 2) == 3
