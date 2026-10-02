import pytest

from mlkit import errors


def test_run_main_turns_data_check_into_exit_2():
    def main():
        raise errors.DataCheckError("too few rows")

    with pytest.raises(SystemExit) as exc:
        errors.run_main(main)
    assert exc.value.code == errors.EXIT_DATA_CHECK == 2
