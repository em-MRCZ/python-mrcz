from mrcz.ioMRC import (
    # Public camelCase API, unchanged for backward compatibility.
    readMRC, writeMRC, asyncReadMRC, asyncWriteMRC,
    readMRCHeader, writeMRCHeader, defaultHeader, setDefaultThreads,
    _setAsyncWorkers, _asyncExecutor,
    # snake_case aliases bound to the same objects.
    read_mrc, write_mrc, async_read_mrc, async_write_mrc,
    read_mrc_header, write_mrc_header, default_header, set_default_threads,
    # Backend availability flags, useful for skipping compressed code paths.
    BLOSC_PRESENT, BLOSC2_PRESENT, CAN_COMPRESS,
)
from mrcz.ioDM import readDM4, asyncReadDM4, read_dm4, async_read_dm4
from mrcz.__version__ import __version__

__all__ = [
    '__version__',
    'readMRC', 'writeMRC', 'asyncReadMRC', 'asyncWriteMRC',
    'readMRCHeader', 'writeMRCHeader', 'defaultHeader', 'setDefaultThreads',
    'read_mrc', 'write_mrc', 'async_read_mrc', 'async_write_mrc',
    'read_mrc_header', 'write_mrc_header', 'default_header', 'set_default_threads',
    'readDM4', 'asyncReadDM4', 'read_dm4', 'async_read_dm4',
    'BLOSC_PRESENT', 'BLOSC2_PRESENT', 'CAN_COMPRESS',
    'test',
]


def test(verbosity: int = 2) -> int:
    """
    Run the ``pytest`` suite for the ``mrcz`` package.

    `pytest` is imported lazily so that it is only needed to run the tests, not
    to import the package.

    Returns
    -------
    The ``pytest`` exit code, where ``0`` means every test passed.
    """
    import os.path

    import pytest

    args = [os.path.dirname(__file__)]
    if verbosity >= 2:
        args.append('-v')
    elif verbosity <= 0:
        args.append('-q')
    return int(pytest.main(args))
