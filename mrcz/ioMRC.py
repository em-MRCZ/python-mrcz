r'''
Conventional MRC2014 and on-the-fly compressed MRCZ file interface

CCPEM MRC2014 specification:
http://www.ccpem.ac.uk/mrc_format/mrc2014.php

IMOD specification:
http://bio3d.colorado.edu/imod/doc/mrc_format.txt

Testing:
http://emportal.nysbc.org/mrc2014/

Tested output on: Gatan GMS, IMOD, Chimera, Relion, MotionCorr, UnBlur
'''

import logging
import os
import os.path
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import Any, BinaryIO

import numpy as np
from packaging.version import Version

from mrcz.__version__ import __version__

logger = logging.getLogger('MRCZ')

# The two backends are not interchangeable. `blosc` emits version-2 chunks, the
# only ones older `mrcz` can read, so it is the write default. `blosc2` emits
# version-5 chunks but reads both, so it is preferred for reading.
try:
    import blosc2
    BLOSC2_PRESENT = True
except ImportError:
    BLOSC2_PRESENT = False

try:
    import blosc
    BLOSC_PRESENT = True
    # For async operations we want to release the GIL in blosc operations and
    # file IO operations.
    blosc.set_releasegil(True)
    DEFAULT_N_THREADS = blosc.detect_number_of_cores()
except ImportError:
    # Can be ImportError or ModuleNotFoundError depending on the Python version,
    # but ModuleNotFoundError is a child of ImportError and is still caught.
    BLOSC_PRESENT = False
    if BLOSC2_PRESENT:
        blosc2.set_releasegil(True)
        DEFAULT_N_THREADS = blosc2.detect_number_of_cores()
    else:
        DEFAULT_N_THREADS = 1
        logger.info('`blosc` meta-compression library not found, file compression disabled.')

# True when a compressed MRCZ file can be read or written at all.
CAN_COMPRESS = BLOSC_PRESENT or BLOSC2_PRESENT

try:
    import rapidjson as json
except ImportError:
    import json
    logger.info('`python-rapidjson` not found, using builtin `json` instead.')


def _defaultMetaSerialize(value: Any) -> Any:
    """
    Is called by `json.dumps()` whenever it encounters an object it does
    not know how to serialize. Currently handles:

    1. Any object with a `serialize()` method, which is assumed to be a helper
       method.
    1. `numpy` scalars as well as `ndarray`
    2. Python `Enum` objects which are serialized as strings in the form
       ``'Enum.{object.__class__.__name__}.{object.name}'``. E.g. ``'Enum.Axis.X'``.
    """
    if hasattr(value, 'serialize'):
        return value.serialize()
    elif hasattr(value, '__array_interface__'):
        # Checking for '__array_interface__' also sanitizes numpy scalars
        # like np.float32 or np.int32
        return value.tolist()
    elif isinstance(value, Enum):
        return f'Enum.{value.__class__.__name__}.{value.name}'
    else:
        raise TypeError(f'Unhandled type for JSON serialization: {type(value)}')

# Do we also want to convert long lists to np.ndarrays?

# Buffer for file I/O
# Quite arbitrary, in bytes (hand-optimized)
BUFFERSIZE = 2**20
BLOSC_BLOCK = 2**16
DEFAULT_HEADER_LEN = 1024

# ENUM dicts for our various Python to MRC constant conversions.
# 0-6 are the blosc1 codecs; 7+ exist only in blosc2. MODE is advisory per the
# MRCZ spec, the authoritative codec is recorded inside the blosc chunk.
COMPRESSOR_ENUM = {0:None, 1:'blosclz', 2:'lz4', 3:'lz4hc', 4:'snappy', 5:'zlib', 6:'zstd',
                   7:'ndlz', 8:'zfp_acc', 9:'zfp_prec', 10:'zfp_rate',
                   11:'openhtj2k', 12:'grok', 13:'openzl', 14:'j2k', 15:'htj2k'}
# `lz4hc` was 2 before 0.6.0, contradicting COMPRESSOR_ENUM, so those files
# report themselves as lz4. Pixel data was unaffected.
REVERSE_COMPRESSOR_ENUM = {name: value for value, name in COMPRESSOR_ENUM.items()}

# Queried rather than hard-coded. `snappy` is gone from c-blosc1 but keeps its
# number so old files still name their codec.
BLOSC1_CODECS = frozenset(blosc.compressor_list()) if BLOSC_PRESENT else frozenset()
BLOSC2_ONLY_CODECS = frozenset(name for name in REVERSE_COMPRESSOR_ENUM
                               if name is not None and name not in
                               {'blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd'})

# Chunk header byte 0 is the format version: 2 for blosc1, 5 for blosc2.
BLOSC_FORMAT_BY_VERSION = {2: 1, 5: 2}

MRC_COMP_RATIO = 1000
CCPEM_ENUM = {0: 'i1', 1:'i2', 2:'f4', 4:'c8', 6:'u2', 7:'i4', 8:'u4', 101:'u1'}
EMAN2_ENUM = {1: 'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4', 7:'f4'}
REVERSE_CCPEM_ENUM = {'int8':0, 'i1':0,
                      'uint4':101,
                      'int16':1, 'i2':1,
                      'uint16':6, 'u2':6,
                      'int32':7, 'i4':7,
                      'uint32': 8, 'u4':8,
                      'float64':2, 'f8':2, 'float32':2, 'f4':2,
                      'complex128':4, 'c16':4, 'complex64':4, 'c8':4}
WARNED_ABOUT_CASTING_F64  = False
WARNED_ABOUT_CASTING_C128 = False
WARNED_ABOUT_BLOSC2_WRITE = False
WARNED_ABOUT_ZFP = False

# Executor for calls to asyncReadMRC and asyncWriteMRC
_asyncExecutor = ThreadPoolExecutor(max_workers=1)


def _setAsyncWorkers(N_workers: int) -> None:
    '''
    **This function is protected as there appears to be little value in using more
    than one worker. It may be subject to removal in the future.**

    Sets the maximum number of background workers that can be used for reading
    or writing with the functions.  Defaults to 1. Generally when writing
    to hard drives use 1 worker. For random-access drives there may be
    advantages to using multiple workers.

    Some test results, 30 files, each 10 x 2048 x 2048 x float-32, on a CPU
    with 4 physical cores:

        HD, 1 worker:   42.0 s
        HD, 2 workers:  50.0 s
        HD, 4 workers:  50.4 s
        SSD, 1 worker:  12.6 s
        SSD, 2 workers: 11.6 s
        SSD, 4 workers:  8.9 s
        SSD, 8 workers: 16.9 s

    Parameters
    ----------
    N_workers
        The number of threads for asynchronous reading and writing to disk
    '''
    if N_workers <= 0:
        raise ValueError('N_workers must be greater than 0')
    if _asyncExecutor._max_workers == N_workers:
        return
    _asyncExecutor._max_workers = N_workers
    _asyncExecutor._adjust_thread_count()


def setDefaultThreads(n_threads: int) -> None:
    """
    Set the default number of threads, if the argument is not provided in
    calls to `readMRC` and `writeMRC`.

    Generally optimal thread count is the number of physical cores, but
    blosc defaults to the number of virtual cores. Therefore on machines with
    hyperthreading it can be more efficient to manually set this value
    """
    global DEFAULT_N_THREADS
    DEFAULT_N_THREADS = int(n_threads)


def _setBloscThreads(n_threads: int) -> None:
    """
    Applies the thread count to whichever backends are present. Both are
    configured because a single call may decompress with `blosc2` while
    compressing with `blosc`.
    """
    if BLOSC_PRESENT:
        blosc.set_nthreads(int(n_threads))
    if BLOSC2_PRESENT:
        blosc2.set_nthreads(int(n_threads))


def _setBloscBlocksize(block_size: int) -> None:
    """
    Blocksize is a global in both backends and affects the bytes emitted, so it
    goes to whichever one will compress.
    """
    if BLOSC_PRESENT:
        blosc.set_blocksize(int(block_size))
    else:
        blosc2.set_blocksize(int(block_size))


def _resolveBackend(compressor: str | None, backend: str | None) -> str | None:
    """
    Picks the compressing library, refusing combinations that cannot produce a
    correctly labelled file. Defaults to blosc1 for interoperability, so blosc2
    must be asked for unless it is the only backend installed.
    """
    global WARNED_ABOUT_BLOSC2_WRITE

    if compressor is None:
        return None
    if backend not in (None, 'blosc1', 'blosc2'):
        raise ValueError(f"backend must be None, 'blosc1' or 'blosc2', got {backend!r}")

    if compressor not in REVERSE_COMPRESSOR_ENUM:
        raise ValueError(f'Unknown compressor {compressor!r}, '
                         f'expected one of {sorted(n for n in REVERSE_COMPRESSOR_ENUM if n)}')

    if backend == 'blosc1':
        if not BLOSC_PRESENT:
            raise ImportError("backend='blosc1' requires the `blosc` package.")
        if compressor not in BLOSC1_CODECS:
            raise ValueError(f'{compressor!r} is not available in `blosc`; this codec needs '
                             f"backend='blosc2'. `blosc` offers {sorted(BLOSC1_CODECS)}.")
        return 'blosc1'

    if backend == 'blosc2':
        if not BLOSC2_PRESENT:
            raise ImportError("backend='blosc2' requires the `blosc2` package.")
        return 'blosc2'

    # Prefer blosc1, its output is the interoperable one.
    if BLOSC_PRESENT and compressor in BLOSC1_CODECS:
        return 'blosc1'
    if not BLOSC2_PRESENT:
        raise ImportError(f'{compressor!r} requires the `blosc2` package.')

    if not WARNED_ABOUT_BLOSC2_WRITE:
        why = (f'{compressor} is a blosc2-only codec' if compressor in BLOSC2_ONLY_CODECS
               else '`blosc` is not installed')
        logger.warning(f'{why}, so this file gets version-5 chunks, unreadable by `mrcz < 0.6`. '
                       'Further warnings suppressed.')
        WARNED_ABOUT_BLOSC2_WRITE = True
    return 'blosc2'


def _bloscCompress(frame: np.ndarray, typesize: int, clevel: int, cname: str,
                   backend: str, blocksize: int = 0, n_threads: int = 1) -> bytes:
    """
    Compresses one frame into a single blosc chunk.

    Uses `compress2` rather than the `blosc2.compress` shim, which silently
    falls back to zstd for an unavailable plugin codec and would write a file
    whose header names a codec the chunk does not hold.
    """
    if not (frame.flags['C_CONTIGUOUS'] and frame.flags['ALIGNED']):
        # Both backends reject non-contiguous arrays.
        frame = np.ascontiguousarray(frame)

    if backend == 'blosc1':
        return blosc.compress(frame, typesize, clevel=clevel,
                              shuffle=blosc.BITSHUFFLE, cname=cname)

    try:
        codec = getattr(blosc2.Codec, cname.upper())
    except AttributeError:
        raise ValueError(f'`blosc2` has no codec named {cname!r}') from None

    global WARNED_ABOUT_ZFP
    if cname.startswith('zfp') and not WARNED_ABOUT_ZFP:
        # zfp needs the shape and itemtype that blosc2 NDArrays carry; MRCZ
        # compresses one flat frame per chunk, leaving it nothing to work with.
        logger.warning(f'Codec {cname!r} needs blosc2 NDArray metadata that MRCZ chunks lack, '
                       'so it will barely compress. Prefer `zstd`. Further warnings suppressed.')
        WARNED_ABOUT_ZFP = True

    try:
        return blosc2.compress2(frame, codec=codec, clevel=clevel,
                                filter=blosc2.Filter.BITSHUFFLE, typesize=typesize,
                                blocksize=blocksize, nthreads=n_threads)
    except RuntimeError as e:
        # c-blosc2 reports a bare "Could not compress the data" for a missing plugin.
        hint = f' Codec {cname!r} may need the `blosc2-{cname}` package installed.' \
               if cname in BLOSC2_ONLY_CODECS else ''
        raise RuntimeError(f'`blosc2` could not compress with codec {cname!r}: {e}.{hint}') from e


def _bloscDecompress(data: bytes) -> bytes:
    """
    Decompresses a single blosc chunk. `blosc2` is preferred because it reads
    both chunk formats; `blosc` reads only version-2.
    """
    if BLOSC2_PRESENT:
        return blosc2.decompress(data)
    if data and data[0] != 2:
        raise ImportError('This file holds blosc2-format (version-5) chunks, which the `blosc` '
                          'package cannot read. Install `blosc2`.')
    return blosc.decompress(data)


def defaultHeader() -> dict:
    r'''
    Generator function to create a metadata header dictionary with the relevant
    fields.

    Returns
    -------
    header
        a default MRC header dictionary with all fields with default values.
    '''
    header = {}
    header['fileConvention'] = 'ccpem'
    header['endian'] = 'le'
    header['MRCtype'] = 0
    header['dimensions'] = np.array([0, 0, 0], dtype=int)
    header['dtype'] = 'u1'

    header['compressor'] = None
    header['backend'] = None
    header['packedBytes'] = 0
    header['clevel'] = 1

    header['maxImage'] = 1.0
    header['minImage'] = 0.0
    header['meanImage'] = 0.0

    header['pixelsize'] = 0.1
    header['pixelunits'] = 'nm' # Can be '\\AA' for Angstroms
    header['voltage'] = 300.0 # kV
    header['C3'] = 2.7 # mm
    header['gain'] = 1.0 # counts/electron

    if CAN_COMPRESS:
        header['n_threads'] = DEFAULT_N_THREADS

    return header


def _getMRCZVersion(label: str | bytes) -> Version | None:
    """
    Checks to see if the first label holds the MRCZ version information,
    in which case it returns a version object. Generally used to recover nicely
    in case of backward compatibility problems.

    Returns
    -------
    version
        returns ``None`` if `label` cannot be parsed.
    """
    if isinstance(label, bytes):
        label = label.decode()

    label = label.rstrip(' \t\r\n\0')
    if not label.startswith('MRCZ'):
        return None

    label = label[4:]
    try:
        return Version(label)
    except ValueError:
        return None


def readMRC(MRCfilename: str | os.PathLike, idx: int | tuple[int, int] | None = None, endian: str = 'le',
            pixelunits: str = '\\AA', fileConvention: str = 'ccpem', useMemmap: bool = False,
            n_threads: int | None = None,
            slices: int | None = None) -> tuple[np.ndarray | list[np.ndarray], dict]:
    r'''
    Imports an MRC/Z file as a NumPy array and a meta-data dict.

    Parameters
    ----------
    image
        a 1-3 dimension ``numpy.ndarray`` with one of the supported data types in
        ``mrcz.REVERSE_CCPEM_ENUM``
    meta
        a ``dict`` with various fields relating to the MRC header information.
        Can also hold arbitrary meta-data, but the use of large numerical data
        is not recommended as it is encoded as text via JSON.
    idx
        Index tuple ``(first, last)`` where first (inclusive) and last (not
        inclusive) indices of images to be read from the stack. Index of first image
        is 0. Negative indices can be used to count backwards. A singleton integer
        can be provided to read only one image. If omitted, will read whole file.
        Compression is currently not supported with this option.
    pixelunits
        can be ``'\\AA' (Angstoms), 'nm', '\mum', or 'pm'``.  Internally pixel
        sizes are always encoded in Angstroms in the MRC file.
    fileConvention
        can be ``'ccpem'`` (equivalent to IMOD) or ``'eman2'``, which is
        only partially supported at present.
    endian
        can be big-endian as ``'be'`` or little-endian as ``'le'``. Defaults
        to `'le'` as the vast majority of modern computers are little-endian.
    n_threads
        is the number of threads to use for decompression, defaults to
        use all virtual cores.
    useMemmap
        returns a ``numpy.memmap`` instead of a ``numpy.ndarray``. Not recommended
        as it will not work with compression.
    slices
        Reflects the number of slices per frame. For example, in time-series
        with multi-channel STEM, would be ``4`` for a 4-quadrant detector. Data
        is always written contiguously in MRC, but will be returned as a list of
        ``[slices, *shape]``-shaped arrays. The default option ``None`` will
        check for a ``'slices'`` field in the meta-data and use that, otherwise
        it defaults to ``0`` which is one 3D array.

    Returns
    -------
    image
        If ``slices == 0`` then a monolithic array is returned, else a ``list``
        of ``[slices, *shape]``-shaped arrays.
    meta
        the stored meta-data in a dictionary. Note that arrays are generally
        returned as lists due to the JSON serialization.

    Example
    -------
        [image, meta] = readMRC(MRCfilename, idx=None,
              pixelunits='\\AA', useMemmap=False, n_threads=None)
    '''

    with open(MRCfilename, 'rb', buffering=BUFFERSIZE) as f:
        # Read in header as a dict

        header, slices = readMRCHeader(MRCfilename, slices, endian=endian,
                                       fileConvention=fileConvention, pixelunits=pixelunits)
        # Support for compressed data in MRCZ

        if ((header['compressor'] in REVERSE_COMPRESSOR_ENUM)
                and (REVERSE_COMPRESSOR_ENUM[header['compressor']] > 0)
                and idx is None):
            return __MRCZImport(f, header, slices, endian=endian, fileConvention=fileConvention,
                                n_threads=n_threads)
        # Else load as uncompressed MRC file

        if idx is not None:
            # If specific images were requested:
            # TO DO: add support to read all images within a range at once

            if header['compressor'] is not None:
                raise RuntimeError('Reading from arbitrary positions not supported for compressed '
                                   f"files. Compressor = {header['compressor']}")
            if np.isscalar(idx):
                indices = np.array([idx, idx], dtype='int')
            else:
                indices = np.array(idx, dtype='int')

            # Convert to old way:
            idx = indices[0]
            n = indices[1] - indices[0] + 1

            if idx < 0:
                # Convert negative index to equivalent positive index:
                idx = header['dimensions'][0] + idx

            # Just check if the desired image is within the stack range:
            if idx < 0 or idx >= header['dimensions'][0]:
                raise ValueError('Error: image or slice index out of range. '
                                 f"idx = {idx}, z_dimension = {header['dimensions'][0]}")
            elif idx + n > header['dimensions'][0]:
                raise ValueError('Error: image or slice index out of range. '
                                 f"idx + n = {idx + n}, z_dimension = {header['dimensions'][0]}")
            elif n < 1:
                raise ValueError(f'Error: n must be >= 1. n = {n}')
            else:
                # We adjust the dimensions of the returned image in the header:
                header['dimensions'][0] = n

                # This offset will be applied to f.seek():
                offset = idx * np.prod(header['dimensions'][1:]) * np.dtype(header['dtype']).itemsize

        else:
            offset = 0

        data_offset = DEFAULT_HEADER_LEN + header['extendedBytes'] + offset
        f.seek(data_offset)
        if bool(useMemmap):
            # `np.memmap` ignores the handle position, so without `offset` it
            # returns the header as pixel data.
            image = np.memmap(f, dtype=header['dtype'],
                              mode='c',
                              offset=data_offset,
                              shape=tuple(int(dim) for dim in header['dimensions']))
        else: # Load entire file into memory
            dims = header['dimensions']
            if slices > 0: # List of NumPy 2D-arrays
                frame_size = slices * np.prod(dims[1:])
                n_frames = dims[0] // slices
                dtype = header['dtype']

                # np.fromfile advances the file pointer `f` for us.
                image = []
                for I in range(n_frames):
                    buffer = np.fromfile(f, dtype=dtype, count=frame_size)
                    buffer = buffer.reshape((slices, dims[1], dims[2])).squeeze()
                    image.append(buffer)

            else: # monolithic NumPy ndarray
                image = np.fromfile(f, dtype=header['dtype'], count=np.prod(dims))

                if header['MRCtype'] == 101:
                    # 4-bit packs two pixels per byte: even in the low nibble.
                    interlaced_image = image
                    image = np.empty(np.prod(dims), dtype=header['dtype'])
                    image[0::2] = np.bitwise_and(interlaced_image, 0x0F)
                    image[1::2] = np.right_shift(interlaced_image, 4)

                image = np.squeeze(image.reshape(dims))

        return image, header


def __MRCZImport(f: BinaryIO, header: dict, slices: int, endian: str = 'le',
                 fileConvention: str = 'ccpem', returnHeader: bool = False,
                 n_threads: int | None = None) -> tuple[np.ndarray | list[np.ndarray], dict]:
    '''
    Equivalent to MRCImport, but for compressed data using the blosc library.

    The following compressors are recommended: [``'zlib'``, ``'zstd'``, ``'lz4'``]

    Memory mapping is not possible in this case at present. Possibly support can
    be added for memory mapping with `c-blosc2`.
    '''
    if not CAN_COMPRESS:
        raise ImportError('Neither `blosc` nor `blosc2` is installed, cannot decompress file.')

    if n_threads is None:
        _setBloscThreads(DEFAULT_N_THREADS)
    else:
        _setBloscThreads(n_threads)

    dims = header['dimensions']
    dtype = header['dtype']
    if slices > 0:
        image = []
        n_frames = dims[0] // slices
    else:
        image = np.empty(dims, dtype=dtype)
        n_frames = dims[0]

    if slices > 1:
        target_shape = (slices, dims[1], dims[2])
    else:
        target_shape = (dims[1], dims[2])
    blosc_chunk_pos = DEFAULT_HEADER_LEN + header['extendedBytes']

    # NOTE: each channel of each frame is separately compressed by blosc,
    # so that if slices is not what was originally input, each slice can
    # be decompressed individually.
    if slices == 1: # List of 2D frames
        for J in range(n_frames):
            f.seek(blosc_chunk_pos)
            ((nbytes, blockSize, ctbytes), (ver_info)) = readBloscHeader(f)
            f.seek(blosc_chunk_pos)
            image.append(np.frombuffer(_bloscDecompress(f.read(ctbytes)),
                                       dtype=dtype).reshape(target_shape))
            blosc_chunk_pos += ctbytes
    elif slices > 1: # List of 3D frames
        for J in range(n_frames):
            frame = np.empty(target_shape, dtype=dtype)
            for I in range(slices):
                f.seek(blosc_chunk_pos)
                ((nbytes, blockSize, ctbytes), (ver_info)) = readBloscHeader(f)
                f.seek(blosc_chunk_pos)
                frame[I,:,:] = np.frombuffer(_bloscDecompress(f.read(ctbytes)),
                                             dtype=dtype).reshape(target_shape[1:])
                blosc_chunk_pos += ctbytes
            image.append(frame)
    else: # Monolithic frame
        for J in range(n_frames):
            f.seek(blosc_chunk_pos)
            ((nbytes, blockSize, ctbytes), (ver_info)) = readBloscHeader(f)
            f.seek(blosc_chunk_pos)
            image[J,:,:] = np.frombuffer(_bloscDecompress(f.read(ctbytes)),
                                         dtype=dtype).reshape(target_shape)
            blosc_chunk_pos += ctbytes

    if header['MRCtype'] == 101:
        # 4-bit packs two pixels per byte: even in the low nibble.
        if slices > 0:
            raise NotImplementedError('MRC type 101 (uint4) not supported with return as `list`')
        interlaced_image = image.ravel()

        image = np.empty(np.prod(header['dimensions']), dtype=dtype)
        # Bit-and and bit-shift to seperate decimated pixels
        image[0::2] = np.bitwise_and(interlaced_image, 0x0F)
        image[1::2] = np.right_shift(interlaced_image, 4)

    if not slices > 0:
        image = np.squeeze(image)

    return image, header


def readBloscHeader(filehandle: BinaryIO) -> tuple[list[int], list[int]]:
    '''
    Reads in the 16 byte header file from a blosc chunk. Blosc header format
    for each chunk is as follows::

        |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
        ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
        |   |   |   |
        |   |   |   +--typesize
        |   |   +------flags
        |   +----------versionlz
        +--------------version

    The layout of the fields read here is identical in the blosc1 (version 2)
    and blosc2 (version 5) chunk formats, so chunk-walking works for both.
    '''
    [version, versionlz, flags, typesize] = np.fromfile(filehandle, dtype='uint8', count=4)
    [nbytes, blocksize, ctbytes] = np.fromfile(filehandle, dtype='uint32', count=3)
    return ([nbytes, blocksize, ctbytes], [version, versionlz, flags, typesize])


def readMRCHeader(MRCfilename: str | os.PathLike, slices: int | None = None, endian: str = 'le',
                  fileConvention: str = 'ccpem', pixelunits: str = '\\AA') -> tuple[dict, int]:
    r'''
    Reads in the first 1024 bytes from an MRC file and parses it into a Python
    dictionary, yielding header information. This function is not intended to be
    called by the user under typical usage.

    Parameters
    ----------
    As per `readMRC`

    Returns
    -------
    header
        All found meta-data in the header and extended header packaged into
        a dictionary.
    slices
        The number of z-slices per frame, as stored in the file.
    '''
    if endian == 'le':
        endchar = '<'
    else:
        endchar = '>'
    dtype_i4 = np.dtype(endchar + 'i4')
    dtype_f4 = np.dtype(endchar + 'f4')

    header = {}
    with open(MRCfilename, 'rb') as f:
        # Grab version information early
        f.seek(224)
        mrcz_version = _getMRCZVersion(f.read(80))

        # Dimensions are stored as [nx, ny, nz] and returned as [nz, ny, nx]
        f.seek(0)
        raw_dimensions = np.fromfile(f, dtype=dtype_i4, count=3)
        raw_mrctype = np.fromfile(f, dtype=dtype_i4, count=1)

        # Hack to fix lack of standard endian indication in the file header
        if int(raw_mrctype[0]) > 16000000:
            # Endianess is backward. `.view()` re-decodes the same bytes;
            # `.byteswap()` would rewrite an already-misdecoded value.
            endchar = '>' if endchar == '<' else '<'
            dtype_i4 = np.dtype(endchar + 'i4')
            dtype_f4 = np.dtype(endchar + 'f4')
            raw_dimensions = raw_dimensions.view(dtype_i4)
            raw_mrctype = raw_mrctype.view(dtype_i4)

        header['dimensions'] = raw_dimensions[::-1]
        header['MRCtype'] = int(raw_mrctype[0])

        # Extract compressor from dtype > MRC_COMP_RATIO
        compressor_id = header['MRCtype'] // MRC_COMP_RATIO
        try:
            header['compressor'] = COMPRESSOR_ENUM[compressor_id]
        except KeyError:
            raise ValueError(f'Error: unrecognized MRCZ compressor id = {compressor_id} '
                             f'in file {MRCfilename}. Known ids: '
                             f'{sorted(COMPRESSOR_ENUM)}') from None
        header['MRCtype'] = header['MRCtype'] % MRC_COMP_RATIO
        logger.debug(f"compressor: {header['compressor']}, MRCtype: {header['MRCtype']}")

        fileConvention = fileConvention.lower()

        if fileConvention == 'eman2':
            try:
                header['dtype'] = EMAN2_ENUM[header['MRCtype']]
            except KeyError:
                raise ValueError(f"Error: unrecognized EMAN2-MRC data type = {header['MRCtype']}")

        elif fileConvention == 'ccpem': # Default is CCPEM
            try:
                header['dtype'] = CCPEM_ENUM[header['MRCtype']]
            except KeyError:
                raise ValueError(f"Error: unrecognized CCPEM-MRC data type = {header['MRCtype']}")
        else:
            raise ValueError(f'Error: unrecognized MRC file convention: {fileConvention}')

        # Apply endian-ness to NumPy dtype
        header['dtype'] = endchar + header['dtype']

        # slices is z-axis per frame for list-of-arrays representation

        if slices is None:
            # We had a bug in version <= 0.4.1 where we wrote the dimensions
            # into both (Nx, Ny, Nz) AND (Mx, My, Mz), therefore the slicing
            # is essentially unknown (and wrong). So we have this version
            # check where we force slices to be 1 (i.e. we assume it is a
            # stack of 2D images).
            if mrcz_version is not None and mrcz_version < Version('0.5.0'):
                logger.warning(f'MRCZ version < 0.5.0 for file {MRCfilename}, assuming slices == 1.')
                slices = 1
            else:
                f.seek(36)
                slices = int(np.fromfile(f, dtype=dtype_i4, count=1)[0])

        # Read in pixelsize
        f.seek(40)
        cellsize = np.fromfile(f, dtype=dtype_f4, count=3)
        header['pixelsize'] = cellsize[::-1] / header['dimensions']
        # MRC is Angstroms by convention
        header['pixelunits'] = pixelunits

        if header['pixelunits'] == '\\AA':
            pass
        elif header['pixelunits'] == '\\mum':
            header['pixelsize'] *= 1E-5
        elif header['pixelunits'] == 'pm':
            header['pixelsize'] *= 100.0
        else: # Default to nm
            header['pixelsize'] *= 0.1

        # Read in [X,Y,Z] array ordering
        # Currently I don't use this
        # f.seek(64)
        # axesTranpose = np.fromfile(f, dtype=endchar + 'i4', count=3) - 1

        # Read in statistics
        f.seek(76)
        header['minImage'], header['maxImage'], header['meanImage'] = np.fromfile(f, dtype=dtype_f4,
                                                                                 count=3)

        # Size of meta-data
        f.seek(92)
        header['extendedBytes'] = int(np.fromfile(f, dtype=dtype_i4, count=1)[0])
        if header['extendedBytes'] > 0:
            f.seek(104)
            header['metaId'] = f.read(4)

        # MODE names only the codec. Byte 0 of the first chunk is the format
        # version, which is what decides who can read the file.
        if header['compressor'] is not None:
            f.seek(DEFAULT_HEADER_LEN + header['extendedBytes'])
            version_byte = f.read(1)
            header['bloscFormat'] = (BLOSC_FORMAT_BY_VERSION.get(version_byte[0])
                                     if version_byte else None)
            header['backend'] = (None if header['bloscFormat'] is None
                                 else f"blosc{header['bloscFormat']}")

        # Read in kV, C3, and gain
        f.seek(132)
        microscope_state = np.fromfile(f, dtype=dtype_f4, count=3)
        header['voltage'] = float(microscope_state[0])
        header['C3']      = float(microscope_state[1])
        header['gain']    = float(microscope_state[2])

        # Read in size of packed data
        f.seek(144)
        # `struct.unpack` returns a tuple; take the single element, and honor
        # the file's byte order rather than the machine's.
        header['packedBytes'] = struct.unpack(endchar + 'q', f.read(8))[0]

        # Now read in JSON meta-data if present
        if 'metaId' in header and header['metaId'] == b'json':
            f.seek(DEFAULT_HEADER_LEN)
            meta = json.loads(f.read(header['extendedBytes']).decode('utf-8'))
            for key, value in meta.items():
                if key not in header:
                    header[key] = value
        return header, slices


def writeMRC(input_image: np.ndarray | list[np.ndarray], MRCfilename: str | os.PathLike,
             meta: dict | None = None, endian: str = 'le', dtype: np.dtype | str | None = None,
             pixelsize: list | tuple = [0.1, 0.1, 0.1], pixelunits: str = '\\AA',
             shape: tuple | None = None, voltage: float = 0.0, C3: float = 0.0, gain: float = 1.0,
             compressor: str | None = None, clevel: int = 1, n_threads: int | None = None,
             quickStats: bool = True, idx: int | None = None, backend: str | None = None) -> None:
    r'''
    Write a conventional MRC file, or a compressed MRCZ file to disk.  If
    compressor is ``None``, then backwards compatibility with other MRC libraries
    should be preserved.  Other libraries will not, however, recognize
    the JSON extended meta-data.

    Parameters
    ----------
    input_image
        The image data to write, should be a 1-3 dimension ``numpy.ndarray``
        or a list of 2-dimensional ``numpy.ndarray``s.
    meta
         will be serialized by JSON and written into the extended header. Note
         that ``rapidjson`` (the default) or ``json`` (the fallback) cannot
         serialize all Python objects, so sanitizing ``meta`` to remove non-standard
         library data structures is advisable, including ``numpy.ndarray`` values.
    dtype
        will cast the data before writing it.
    pixelsize
        is [z,y,x] pixel size (singleton values are ok for square/cubic pixels)
    pixelunits
        one of
        - ``'\\AA'`` for Angstroms
        - ``'pm'`` for picometers
        - ``'\mum'`` for micrometers
        - ``'nm'`` for nanometers.
        MRC standard is always Angstroms, so pixelsize is converted internally
        from nm to Angstroms as needed.
    shape
        is only used if you want to later append to the file, such as
        merging together Relion particles for Frealign.  Not recommended and
        only present for legacy reasons.
    voltage
        accelerating potential in keV
    C3
        spherical aberration in mm
    gain
        detector gain in units (counts/primary electron)
    compressor
        is a choice of ``None, 'lz4', 'zlib', 'zstd'``, plus ``'blosclz'``, ``'lz4hc'``
        - ``'lz4'`` is  generally the fastest.
        - ``'zstd'`` generally gives the best compression performance, and is still almost
          as fast as 'lz4' with ``clevel == 1``.
        The blosc2-only codecs (``'ndlz'``, the ``'zfp_*'`` family, ``'openhtj2k'``,
        ``'grok'``, ``'openzl'``, ``'j2k'``, ``'htj2k'``) are also accepted and
        force ``backend='blosc2'``. Most ship as separate plugin distributions
        and raise if absent.
    backend
        one of ``None``, ``'blosc1'`` or ``'blosc2'``. ``None`` picks ``'blosc1'``
        when it is installed and can emit the codec, since its version-2 chunks
        are what older ``mrcz`` can read. ``'blosc2'`` opts in to version-5
        chunks, which ``mrcz < 0.6`` cannot read.
    clevel
        the compression level, 1 is fastest, 9 is slowest. The compression ratio
        will rise slowly with clevel (but not as fast as the write time slows
        down).
    n_threads
        number of threads to use for blosc compression.  Defaults to number of
        virtual cores if ``None``.
    quickStats
        estimates the image mean, min, max from the first frame only, which
        saves computational time for image stacks. Generally strongly advised to
        be ``True``.
    idx
        can be used to write an image or set of images starting at a specific
        position in the MRC file (which may already exist). Index of first image
        is 0. A negative index can be used to count backwards. If omitted, will
        write whole stack to file. If writing to an existing file, compression
        or extended MRC2014 headers are currently not supported with this option.

    Returns
    -------
    ``None``

    Warning
    -------
    MRC definitions are not consistent. Generally we support the CCPEM2014 schema
    as much as possible.
    '''

    if not CAN_COMPRESS and compressor is not None:
        raise ImportError('Neither `blosc` nor `blosc2` is installed, cannot use file compression.')

    # For dask, we don't want to import dask, but we can still work-around how to
    # check its type without isinstance()
    image_type = type(input_image)
    if image_type.__module__ == 'dask.array.core' and image_type.__name__ == 'Array':
        # Ideally it would be faster to iterate over the chunks and pass each one
        # to blosc but that likely requires c-blosc2
        input_image = input_image.__array__()
        dims = input_image.shape

    slices = 0
    global WARNED_ABOUT_CASTING_F64, WARNED_ABOUT_CASTING_C128

    if isinstance(input_image, (tuple, list)):
        shape = input_image[0].shape
        ndim = input_image[0].ndim
        if ndim == 3:
            slices = shape[0]
            shape = shape[1:]
        elif ndim == 2:
            slices = 1
        else:
            raise ValueError('For a sequence of arrays, only 2D or 3D arrays are handled.')

        dims = np.array([len(input_image)*slices, shape[0], shape[1]])

        # Verify that each image in the list is the same 2D shape and dtype
        first_shape = input_image[0].shape
        first_dtype = input_image[0].dtype

        # Cast float64 -> float32, complex128 -> complex64, into a NEW list;
        # casting in place would narrow the caller's own frames.
        frames = list(input_image)
        for J, z_slice in enumerate(frames):
            if z_slice.shape != first_shape:
                raise ValueError(f'Frame {J} has shape {z_slice.shape}, expected {first_shape}')

            if z_slice.dtype == np.float64 or z_slice.dtype == float:
                if not WARNED_ABOUT_CASTING_F64:
                    logger.warning(f'Casting {MRCfilename} to `numpy.float32`, further warnings '
                                   'will be suppressed.')
                    WARNED_ABOUT_CASTING_F64 = True
                frames[J] = z_slice.astype(np.float32)
            elif z_slice.dtype == np.complex128:
                if not WARNED_ABOUT_CASTING_C128:
                    logger.warning(f'Casting {MRCfilename} to `numpy.complex64`, further warnings '
                                   'will be suppressed.')
                    WARNED_ABOUT_CASTING_C128 = True
                frames[J] = z_slice.astype(np.complex64)
            elif z_slice.dtype != first_dtype:
                raise TypeError(f'Frame {J} has dtype {z_slice.dtype}, expected {first_dtype}')
        input_image = frames

    else: # Array-'like' object
        dims = input_image.shape
        if input_image.ndim == 2:
            # If it's a 2D image we force it to 3D - this makes life easier later:
            input_image = input_image.reshape((1, input_image.shape[0], input_image.shape[1]))

        # Cast float64 -> float32, and complex128 -> complex64
        if input_image.dtype == np.float64 or input_image.dtype == float:
            if not WARNED_ABOUT_CASTING_F64:
                logger.warning(f'Casting {MRCfilename} to `numpy.float32`')
                WARNED_ABOUT_CASTING_F64 = True
            input_image = input_image.astype(np.float32)
        elif input_image.dtype == np.complex128:
            if not WARNED_ABOUT_CASTING_C128:
                logger.warning(f'Casting {MRCfilename} to `numpy.complex64`')
                WARNED_ABOUT_CASTING_C128 = True
            input_image = input_image.astype(np.complex64)

    # We will need this regardless if writing to an existing file or not:
    if endian == 'le':
        endchar = '<'
    else:
        endchar = '>'

    # We now check if we have to create a new header (i.e. new file) or not. If
    # the file exists, but idx is 'None', it will be replaced by a new file
    # with new header anyway:
    if os.path.isfile(MRCfilename):
        idxnewfile = idx is None
    else:
        idxnewfile = True

    if idxnewfile:
        if dtype == 'uint4' and compressor is not None:
            raise TypeError('uint4 packing is not compatible with compression, use int8 datatype.')

        header = {'meta': meta}
        if dtype is None:
            if slices > 0:
                header['dtype'] = endchar + input_image[0].dtype.str.lstrip('<>|=')
            else:
                header['dtype'] = endchar + input_image.dtype.str.lstrip('<>|=')
        else:
            header['dtype'] = dtype

        # Now we need to filter dtype to make sure it's actually acceptable to MRC
        if not header['dtype'].strip('<>|') in REVERSE_CCPEM_ENUM:
            raise TypeError(f"ioMRC.MRCExport: Unsupported dtype cast for MRC {header['dtype']}")

        header['dimensions'] = dims

        header['pixelsize'] = pixelsize
        header['pixelunits'] = pixelunits
        header['shape'] = shape

        # This overhead calculation is annoying but many 3rd party tools that use
        # MRC require these statistical parameters.
        if bool(quickStats):
            if slices > 0:
                first_image = input_image[0]
            else:
                first_image = input_image[0,:,:]

            imMin = first_image.real.min(); imMax = first_image.real.max()
            header['maxImage'] = imMax
            header['minImage'] =  imMin
            header['meanImage'] = 0.5*(imMax + imMin)
        else:
            if slices > 0:
                header['maxImage'] = np.max([z_slice.real.max() for z_slice in input_image])
                header['minImage'] = np.min([z_slice.real.min() for z_slice in input_image])
                header['meanImage'] = np.mean([z_slice.real.mean() for z_slice in input_image])
            else:
                header['maxImage'] = input_image.real.max()
                header['minImage'] = input_image.real.min()
                header['meanImage'] = input_image.real.mean()

        header['voltage'] = voltage
        if not bool(header['voltage']):
            header['voltage'] = 0.0
        header['C3'] = C3
        if not bool(header['C3']):
            header['C3'] = 0.0
        header['gain'] = gain
        if not bool(header['gain']):
            header['gain'] = 1.0

        header['compressor'] = compressor
        header['clevel'] = clevel
        header['backend'] = _resolveBackend(compressor, backend)
        if n_threads is None and CAN_COMPRESS:
            n_threads = DEFAULT_N_THREADS
        header['n_threads'] = n_threads

        if dtype == 'uint4':
            if slices > 0:
                raise NotImplementedError('Saving of lists of arrays not supported for '
                                          '`dtype=uint4`')
            # Decimate to packed 4-bit
            input_image = input_image.astype('uint8')
            input_image = input_image[:,:,::2] + np.left_shift(input_image[:,:,1::2], 4)

    else: # We are going to append to an already existing file:
        # So we try to figure out its header with 'CCPEM' or 'eman2' file conventions:
        try:
            header, slices = readMRCHeader(MRCfilename, slices=None, endian=endian,
                                           fileConvention='CCPEM', pixelunits=pixelunits)

        except ValueError:
            try:
                header, slices = readMRCHeader(MRCfilename, slices=None, endian=endian,
                                               fileConvention='eman2', pixelunits=pixelunits)
            except ValueError:
                # If neither 'CCPEM' nor 'eman2' formats satisfy:
                raise ValueError(f'Error: unrecognized MRC type for file: {MRCfilename} ')

        # If the file already exists, its X,Y dimensions must be consistent with the current
        # image to be written:
        if np.any(header['dimensions'][1:] != input_image.shape[1:]):
            raise ValueError('Error: x,y dimensions of image do not match that of MRC file: '
                             f'{MRCfilename} ')
            # TO DO: check also consistency of dtype?

        if 'meta' not in header.keys():
            header['meta'] = meta

    # Now that we have a proper header, we go into the details of writing to a specific position:
    if idx is not None:
        if header['compressor'] is not None:
            raise RuntimeError('Writing at arbitrary positions not supported for compressed files. '
                               f"Compressor = {header['compressor']}")

        idx = int(idx)
        # Force 2D to 3D dimensions:
        if len(header['dimensions']) == 2:
            header['dimensions'] = np.array([1, header['dimensions'][0], header['dimensions'][1]])

        # Convert negative index to equivalent positive index:
        if idx < 0:
            idx = header['dimensions'][0] + idx

        # Just check if the desired image is within the stack range:
        # In principle we could write to a position beyond the limits of the file (missing slots
        # would be filled with zeros), but let's avoid that the user writes a big file with zeros
        # by mistake. So only positions within or immediately consecutive to the stack are allowed:
        if idx < 0 or idx > header['dimensions'][0]:
            raise ValueError('Error: image or slice index out of range. '
                             f"idx = {idx}, z_dimension = {header['dimensions'][0]}")

        # The new Z dimension may be larger than that of the existing file, or even of the new
        # file, if an index larger than the current stack is specified:
        newZ = idx + input_image.shape[0]
        if newZ > header['dimensions'][0]:
            header['dimensions'] = np.array([idx + input_image.shape[0],
                                             header['dimensions'][1], header['dimensions'][2]])

        # This offset will be applied to f.seek():
        offset = idx * np.prod(header['dimensions'][1:]) * np.dtype(header['dtype']).itemsize

    else:
        offset = 0

    __MRCExport(input_image, header, MRCfilename, slices,
                endchar=endchar, offset=offset, idxnewfile=idxnewfile)


def __MRCExport(input_image: np.ndarray | list[np.ndarray], header: dict,
                MRCfilename: str | os.PathLike, slices: int, endchar: str = '<',
                offset: int = 0, idxnewfile: bool = True) -> None:
    '''
    MRCExport private interface with a dictionary rather than a mess of function
    arguments.
    '''

    if idxnewfile: # If forcing a new file we truncate it even if it already exists:
        fmode = 'wb'
    else: # Otherwise we'll just update its header and append images as required:
        fmode = 'rb+'

    with open(MRCfilename, fmode, buffering=BUFFERSIZE) as f:
        extendedBytes = writeMRCHeader(f, header, slices, endchar=endchar)
        f.seek(DEFAULT_HEADER_LEN + extendedBytes + offset)

        dtype = header['dtype']
        if ('compressor' in header) \
                and (header['compressor'] in REVERSE_COMPRESSOR_ENUM) \
                and (REVERSE_COMPRESSOR_ENUM[header['compressor']]) > 0:
            # compressed MRCZ
            logger.debug(f"Compressing {MRCfilename} with compressor "
                         f"{header['compressor']}{header['clevel']}")

            applyCast = False
            if slices > 0:
                chunkSize = input_image[0].size
                typeSize = input_image[0].dtype.itemsize
                if dtype != 'uint4' and input_image[0].dtype != dtype:
                    applyCast = True
            else:
                chunkSize = input_image[0,:,:].size
                typeSize = input_image.dtype.itemsize
                if dtype != 'uint4' and input_image.dtype != dtype:
                    applyCast = True

            _setBloscThreads(header['n_threads'])
            # for small image dimensions we need to scale blocksize appropriately
            # so we use the available cores
            block_size = int(min(BLOSC_BLOCK, chunkSize // header['n_threads']))
            _setBloscBlocksize(block_size)

            header['packedBytes'] = 0

            clevel = header['clevel']
            cname = header['compressor']
            # An `idx` append or a hand-built header will not carry a backend.
            backend = header.get('backend') or _resolveBackend(cname, None)

            # For 3D frames in lists, we need to further sub-divide each frame
            # into slices so that each channel is compressed seperately by
            # blosc.
            if slices > 1:
                deep_image = input_image # grab a reference
                input_image = []
                for frame in deep_image:
                    for I in range(slices):
                        input_image.append(frame[I,:,:])

            for J, frame in enumerate(input_image):
                if applyCast:
                    frame = frame.astype(dtype)

                compressedData = _bloscCompress(frame, typeSize, clevel, cname, backend,
                                                blocksize=block_size,
                                                n_threads=header['n_threads'])

                f.write(compressedData)
                header['packedBytes'] += len(compressedData)

            # Rewind and write out the total compressed size
            f.seek(144)
            np.array(header['packedBytes'], dtype=endchar + 'i8').tofile(f)

        else: # vanilla MRC
            if slices > 0:
                if dtype != 'uint4' and dtype != input_image[0].dtype:
                    for z_slice in input_image:
                        z_slice.astype(dtype).tofile(f)
                else:
                    for z_slice in input_image:
                        z_slice.tofile(f)
            else:
                if dtype != 'uint4' and dtype != input_image.dtype:
                    input_image = input_image.astype(dtype)
                input_image.tofile(f)


def writeMRCHeader(f: BinaryIO, header: dict, slices: int, endchar: str = '<') -> int:
    r'''
    Writes a header to the file-like object ``f``, requires a dict called
    ``header`` to parse the appropriate fields.

    Returns
    -------
    The length in bytes of the extended (JSON) header, or ``0`` if there is none.

    Note
    ----
    Use `defaultHeader()` to retrieve an example with all potential fields.
    '''
    dtype_f4 = endchar + 'f4'
    dtype_i4 = endchar + 'i4'

    f.seek(0)
    # Write dimensions
    if len(header['dimensions']) == 2: # force to 3-D
        dimensions = np.array([1, header['dimensions'][0], header['dimensions'][1]])
    else:
        dimensions = np.array(header['dimensions'])

    # Flip to Fortran order
    dimensions = dimensions[::-1]
    dimensions.astype(dtype_i4).tofile(f)

    # 64-bit floats are automatically down-cast
    dtype = header['dtype'].lower().strip('<>|')
    try:
        mrc_mode = int(REVERSE_CCPEM_ENUM[dtype])
    except KeyError:
        raise ValueError(f'Warning: Unknown dtype for MRC encountered = {dtype}')

    # Add 1000 * COMPRESSOR_ENUM to the dtype for compressed data
    if ('compressor' in header
                and header['compressor'] in REVERSE_COMPRESSOR_ENUM
                and REVERSE_COMPRESSOR_ENUM[header['compressor']] > 0):
        header['compressor'] = header['compressor'].lower()
        mrc_mode += MRC_COMP_RATIO * REVERSE_COMPRESSOR_ENUM[header['compressor']]

        # How many bytes in an MRCZ file, so that the file can be appended-to.
        # A new header has no `packedBytes` yet; `__MRCExport` writes the real
        # value once known, so the whole compressed file never sits in RAM.
        if 'packedBytes' in header:
            f.seek(144)
            np.array(header['packedBytes'], dtype=endchar + 'i8').tofile(f)

    f.seek(12)
    np.array(mrc_mode, dtype=dtype_i4).tofile(f)

    # Print NXSTART, NYSTART, NZSTART
    np.array([0, 0, 0], dtype=dtype_i4).tofile(f)

    # Print MX, MY, MZ, the sampling. We only allow for slicing along the z-axis,
    # e.g. for multi-channel STEM.
    f.seek(36)
    np.array(slices, dtype=dtype_i4).tofile(f)

    # Print cellsize = pixelsize * dimensions
    if header['pixelunits'] == '\\AA':
        AApixelsize = np.array(header['pixelsize'])
    elif header['pixelunits'] == '\\mum':
        AApixelsize = np.array(header['pixelsize'])*10000.0
    elif header['pixelunits'] == 'pm':
        AApixelsize = np.array(header['pixelsize'])/100.0
    else: # Default is nm
        AApixelsize = np.array(header['pixelsize'])*10.0

    # Flatten first: a shape-(1,) `pixelsize` used to broadcast to (3, 3) and
    # write nine floats over the cell angles and axis associations behind it.
    AApixelsize = np.atleast_1d(AApixelsize).ravel()
    if AApixelsize.size == 1:
        cellsize = np.repeat(AApixelsize, 3) * dimensions
    elif AApixelsize.size == 2:
        # Default to z-axis pixelsize of 10.0 Angstroms
        cellsize = np.array([10.0, AApixelsize[0], AApixelsize[1]])[::-1] * dimensions
    elif AApixelsize.size == 3:
        cellsize = AApixelsize[::-1] * dimensions
    else:
        raise ValueError(f'pixelsize must have 1, 2 or 3 elements, got {AApixelsize.size}')

    f.seek(40)
    np.array(cellsize, dtype=dtype_f4).tofile(f)
    # Print default cell angles
    np.array([90.0, 90.0, 90.0], dtype=dtype_f4).tofile(f)

    # Print axis associations (we use C ordering internally in all Python code)
    np.array([1, 2, 3], dtype=dtype_i4).tofile(f)

    # Print statistics (if available)
    f.seek(76)
    np.array(header.get('minImage', 0.0), dtype=dtype_f4).tofile(f)
    np.array(header.get('maxImage', 1.0), dtype=dtype_f4).tofile(f)
    np.array(header.get('meanImage', 0.0), dtype=dtype_f4).tofile(f)

    # We'll put the compressor info and number of compressed bytes in 132-204
    # and new metadata
    # RESERVED: 132: 136: 140 : 144 for voltage, C3, and gain
    f.seek(132)
    if 'voltage' in header:
        np.array(header['voltage'], dtype=dtype_f4).tofile(f)
    if 'C3' in header:
        np.array(header['C3'], dtype=dtype_f4).tofile(f)
    if 'gain' in header:
        np.array(header['gain'], dtype=dtype_f4).tofile(f)

    # Magic MAP_ indicator that tells us this is in-fact an MRC file
    f.seek(208)
    f.write(b'MAP ')
    # Write a machine stamp, '17,17' for big-endian or '68,65' for little
    # Note that the MRC format doesn't indicate the endianness of the endian
    # identifier...
    f.seek(212)
    if endchar == '<':
        f.write(struct.pack('BB', 68, 65))
    else:
        f.write(struct.pack('BB', 17, 17))

    # Write b'MRCZ<version>' into labels
    f.seek(220)
    # Machine-native here made big-endian files claim 16777216 labels.
    f.write(struct.pack(endchar + 'i', 1)) # We have one label
    f.write(b'MRCZ' + __version__.encode('ascii'))

    # Extended header, if meta is not None
    if isinstance(header['meta'], dict):
        jsonMeta = json.dumps(header['meta'], default=_defaultMetaSerialize).encode('utf-8')

        jsonLen = len(jsonMeta)
        # Length of extended header
        f.seek(92)
        f.write(struct.pack(endchar + 'i', jsonLen))

        # 4-byte char ID string of extended metadata type
        f.seek(104)
        f.write(b'json')

        # Go to the extended header
        f.seek(DEFAULT_HEADER_LEN)
        f.write(jsonMeta)
        return jsonLen

    # No extended header
    return 0


def asyncReadMRC(*args, **kwargs) -> Future:
    '''
    Calls `readMRC` in a separate thread and executes it in the background.

    Parameters
    ----------
    Valid arguments are as for `readMRC()`.

    Returns
    -------
    future
        A ``concurrent.futures.Future()`` object.  Calling ``future.result()``
        will halt until the read is finished and returns the image and meta-data
        as per a normal call to `readMRC`.

    Example
    -------

        worker = asyncReadMRC( 'someones_file.mrc' )
        # Do some work
        mrcImage, mrcMeta = worker.result()
    '''
    return _asyncExecutor.submit(readMRC, *args, **kwargs)


def asyncWriteMRC(*args, **kwargs) -> Future:
    '''
    Calls `writeMRC` in a seperate thread and executes it in the background.

    Parameters
    ----------
    Valid arguments are as for `writeMRC()`.

    Returns
    -------
    future
        A ``concurrent.futures.Future`` object.  If needed, you can call
        ``future.result()`` to wait for the write to finish, or check with
        ``future.done()``. Most of the time you can ignore the return and let
        the system write unmonitored.  An exception would be if you need to pass
        in the output to a subprocess.

    Example
    -------

        worker = asyncWriteMRC( npImageData, 'my_mrcfile.mrc' )
        # Do some work
        if not worker.done():
            time.sleep(0.001)
        # File is written to disk
    '''
    return _asyncExecutor.submit(writeMRC, *args, **kwargs)


# snake_case aliases for the public API. These are bindings to the same objects
# rather than wrappers, so the camelCase names remain fully supported for
# downstream consumers such as `hyperspy`.
read_mrc = readMRC
write_mrc = writeMRC
async_read_mrc = asyncReadMRC
async_write_mrc = asyncWriteMRC
read_mrc_header = readMRCHeader
write_mrc_header = writeMRCHeader
read_blosc_header = readBloscHeader
default_header = defaultHeader
set_default_threads = setDefaultThreads
