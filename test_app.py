"""
Test app
"""

from app import main


def test_main():
    result = main(test = True)
    assert result.exit_code == 0
    assert result.output == "3\n"
