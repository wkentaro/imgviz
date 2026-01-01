import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="show images during tests",
    )


@pytest.fixture
def show(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--show")
