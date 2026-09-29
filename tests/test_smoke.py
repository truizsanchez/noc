import noc
import noc.common


def test_package_imports() -> None:
    assert noc.__doc__
    assert noc.common.__doc__
