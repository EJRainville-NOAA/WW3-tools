import ww3tools


def test_package_imports():
    assert hasattr(ww3tools, "__version__")
    assert hasattr(ww3tools, "data")
    assert hasattr(ww3tools, "interpolation")
    assert hasattr(ww3tools, "validation")
