import math

import pandas as pd
import pytest

from mlkit.labels import forward_direction, forward_return


def test_forward_direction_looks_horizon_rows_ahead():
    close = pd.Series([10.0, 11.0, 9.0, 12.0, 12.0])

    label = forward_direction(close, horizon=2)

    # 10 -> 9 down, 11 -> 12 up, 9 -> 12 up, last two have no future
    assert label.tolist()[:3] == [0.0, 1.0, 1.0]
    assert math.isnan(label.iloc[3]) and math.isnan(label.iloc[4])


def test_equal_close_is_not_up():
    label = forward_direction(pd.Series([5.0, 5.0]), horizon=1)
    assert label.iloc[0] == 0.0


def test_forward_return():
    ret = forward_return(pd.Series([100.0, 110.0]), horizon=1)
    assert ret.iloc[0] == pytest.approx(0.10)
    assert math.isnan(ret.iloc[1])
