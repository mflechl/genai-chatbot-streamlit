"""
Test app
"""

from app import main


def test_main():
    result = main()
    assert result == 0
