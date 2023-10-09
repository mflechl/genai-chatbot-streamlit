"""
Test main
"""

from click.testing import CliRunner
from main import add_cli


def test_main():
    runner = CliRunner()
    result = runner.invoke(add_cli, ["1", "2"])
    assert result.exit_code == 0
    assert result.output == "3\n"
