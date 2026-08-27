# -*- coding: utf-8 -*-
'''
Created on Fri Sep 30 09:44:09 2016

@author: Robert A. McLeod
'''

import logging
import os.path
import struct
from enum import Enum

import numpy as np
import numpy.testing as npt
import pytest

import mrcz

log = logging.getLogger(__name__)

# Angstroms. Doubled because a lone '\A' is an invalid escape sequence; both
# spellings give the same string at runtime.
ANGSTROM = '\\AA'

requires_blosc = pytest.mark.skipif(not mrcz.CAN_COMPRESS,
                                    reason='neither `blosc` nor `blosc2` is installed')

# A compressor is only listed here if the installed backend can actually emit it.
COMPRESSORS = [None]
if mrcz.CAN_COMPRESS:
    COMPRESSORS += ['zstd', 'lz4']


@pytest.fixture
def rng():
    """Seeded so that a failure can be reproduced exactly."""
    return np.random.default_rng(42)


def make_stack(rng, dtype, shape=(2, 128, 96)):
    """Builds test data of the requested dtype without relying on the legacy global RNG."""
    dtype = np.dtype(dtype)
    if dtype.kind == 'c':
        return (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype)
    elif dtype.kind == 'f':
        return rng.normal(size=shape).astype(dtype)
    return rng.integers(0, 10, size=shape).astype(dtype)


#==============================================================================
# ioMRC Test
#
# Internal python-only test. Build a random image and save and re-load it.
#==============================================================================
class TestPythonMrcz:

    def compReadWrite(self, testMage, tmp_path, casttype=None, compressor=None, clevel=1):
        # This is the main functions which reads and writes from disk.
        mrcName = str(tmp_path / 'testMage.mrc')
        pixelsize = np.array([1.2, 2.6, 3.4])

        mrcz.writeMRC(testMage, mrcName, dtype=casttype,
                      pixelsize=pixelsize, pixelunits=ANGSTROM,
                      voltage=300.0, C3=2.7, gain=1.05,
                      compressor=compressor, clevel=clevel)

        rereadMage, rereadHeader = mrcz.readMRC(mrcName, pixelunits=ANGSTROM)

        npt.assert_array_almost_equal(testMage, rereadMage)
        npt.assert_array_almost_equal(rereadHeader['pixelsize'], pixelsize)
        assert rereadHeader['pixelunits'] == ANGSTROM
        npt.assert_almost_equal(rereadHeader['voltage'], 300.0)
        npt.assert_almost_equal(rereadHeader['C3'], 2.7)
        npt.assert_almost_equal(rereadHeader['gain'], 1.05)

    @pytest.mark.parametrize('compressor,clevel', [(None, 1), ('zstd', 1), ('lz4', 9)])
    @pytest.mark.parametrize('dtype', ['float32', 'int8', 'int16', 'uint16', 'complex64'])
    def test_roundtrip(self, rng, tmp_path, dtype, compressor, clevel):
        if compressor is not None and not mrcz.CAN_COMPRESS:
            pytest.skip('neither `blosc` nor `blosc2` is installed')
        log.info(f'Testing {compressor}_{clevel} MRC, {dtype}')
        self.compReadWrite(make_stack(rng, dtype), tmp_path,
                           compressor=compressor, clevel=clevel)

    def test_MRC_uint4(self, rng, tmp_path):
        log.info('Testing uncompressed MRC, uint-4')
        testMage = rng.integers(0, 10, size=(2, 128, 96)).astype('int8')
        self.compReadWrite(testMage, tmp_path, casttype='uint4', compressor=None)

    @requires_blosc
    def test_JSON(self, rng, tmp_path):
        testMage = rng.integers(0, 10, size=(3, 128, 64)).astype('int8')
        meta = {'foo': 5, 'bar': 42}
        mrcName = str(tmp_path / 'testMage.mrcz')

        pixelsize = [1.2, 5.6, 3.4]

        mrcz.writeMRC(testMage, mrcName, meta=meta,
                      pixelsize=pixelsize, pixelunits=ANGSTROM,
                      voltage=300.0, C3=2.7, gain=1.05,
                      compressor='zstd', clevel=1, n_threads=1)

        rereadMage, rereadHeader = mrcz.readMRC(mrcName, pixelunits=ANGSTROM)

        assert testMage.shape == rereadMage.shape
        assert testMage.dtype == rereadMage.dtype
        for key in meta:
            assert meta[key] == rereadHeader[key]

        npt.assert_array_almost_equal(testMage, rereadMage)
        npt.assert_almost_equal(rereadHeader['voltage'], 300.0)
        npt.assert_array_almost_equal(rereadHeader['pixelsize'], pixelsize)
        assert rereadHeader['pixelunits'] == ANGSTROM
        npt.assert_almost_equal(rereadHeader['C3'], 2.7)
        npt.assert_almost_equal(rereadHeader['gain'], 1.05)

    @requires_blosc
    def test_async(self, rng, tmp_path):
        testMage = rng.integers(0, 10, size=(3, 128, 64)).astype('int8')
        meta = {'foo': 5, 'bar': 42}
        mrcName = str(tmp_path / 'testMage.mrcz')

        pixelsize = [1.2, 5.6, 3.4]

        worker = mrcz.asyncWriteMRC(testMage, mrcName, meta=meta,
                                    pixelsize=pixelsize, pixelunits=ANGSTROM,
                                    voltage=300.0, C3=2.7, gain=1.05,
                                    compressor='zstd', clevel=1, n_threads=1)

        worker.result() # Wait for write to finish

        worker = mrcz.asyncReadMRC(mrcName, pixelunits=ANGSTROM)
        rereadMage, rereadHeader = worker.result()

        assert testMage.shape == rereadMage.shape
        assert testMage.dtype == rereadMage.dtype
        for key in meta:
            assert meta[key] == rereadHeader[key]

        npt.assert_array_almost_equal(testMage, rereadMage)
        npt.assert_almost_equal(rereadHeader['voltage'], 300.0)
        npt.assert_array_almost_equal(rereadHeader['pixelsize'], pixelsize)
        assert rereadHeader['pixelunits'] == ANGSTROM
        npt.assert_almost_equal(rereadHeader['C3'], 2.7)
        npt.assert_almost_equal(rereadHeader['gain'], 1.05)

    @pytest.mark.parametrize('compressor', COMPRESSORS)
    @pytest.mark.parametrize('frame_shape', [(32, 16), (3, 32, 32)])
    def test_list_roundtrip(self, rng, tmp_path, frame_shape, compressor):
        # Distinct arrays: `[arr] * 3` tests one array three times and misses
        # frame ordering mistakes.
        testMage = [rng.integers(0, 10, size=frame_shape).astype('int8') for _ in range(3)]
        mrcName = str(tmp_path / 'testMage.mrcz')

        mrcz.writeMRC(testMage, mrcName, pixelsize=[5.6, 3.4],
                      compressor=compressor, clevel=1, n_threads=1)

        rereadMage, _ = mrcz.readMRC(mrcName, pixelunits=ANGSTROM)

        assert isinstance(rereadMage, list)
        assert len(rereadMage) == len(testMage)

        for testFrame, rereadFrame in zip(testMage, rereadMage):
            assert testFrame.dtype == rereadFrame.dtype
            npt.assert_array_almost_equal(testFrame, rereadFrame)

    @pytest.mark.parametrize('compressor', COMPRESSORS)
    @pytest.mark.parametrize('slices', [1, 2])
    def test_list_change_output_shape(self, rng, tmp_path, slices, compressor):
        testMage = rng.integers(0, 10, size=(6, 32, 32)).astype('int8')
        mrcName = str(tmp_path / 'testMage.mrcz')

        mrcz.writeMRC(testMage, mrcName, pixelsize=[5.6, 3.4],
                      compressor=compressor, clevel=1, n_threads=1)

        rereadMage, _ = mrcz.readMRC(mrcName, pixelunits=ANGSTROM, slices=slices)

        assert isinstance(rereadMage, list)
        assert len(rereadMage) == testMage.shape[0] // slices

    @requires_blosc
    def test_strided_array(self, rng, tmp_path):
        log.info('Testing strided array MRC')
        testMage = rng.integers(0, 32, size=(2, 128, 96)).astype(np.int8)[:,::2,::2]
        self.compReadWrite(testMage, tmp_path, compressor='zstd', clevel=1)

    @requires_blosc
    @pytest.mark.parametrize('wide,narrow', [('float64', 'float32'), ('complex128', 'complex64')])
    def test_cast_array(self, rng, tmp_path, wide, narrow):
        log.info(f'Testing {wide} casting')
        wide_mage = make_stack(rng, wide)
        narrow_mage = wide_mage.astype(narrow)

        mrcName = str(tmp_path / 'testMage.mrc')

        mrcz.writeMRC(wide_mage, mrcName, compressor='zstd', clevel=1)
        rereadMage, _ = mrcz.readMRC(mrcName)

        npt.assert_array_almost_equal(narrow_mage, rereadMage)

    @requires_blosc
    @pytest.mark.parametrize('wide,narrow', [('float64', 'float32'), ('complex128', 'complex64')])
    def test_cast_list(self, rng, tmp_path, wide, narrow):
        log.info(f'Testing {wide} list casting')
        wide_mage = [make_stack(rng, wide, shape=(128, 96)) for _ in range(2)]
        narrow_mage = [frame.astype(narrow) for frame in wide_mage]

        mrcName = str(tmp_path / 'testMage.mrc')

        mrcz.writeMRC(wide_mage, mrcName, compressor='zstd', clevel=1)
        rereadMage, _ = mrcz.readMRC(mrcName)

        npt.assert_array_almost_equal(narrow_mage[0], rereadMage[0])
        npt.assert_array_almost_equal(narrow_mage[1], rereadMage[1])

    def test_numpy_metadata(self, tmp_path):
        log.info('Testing NumPy types in meta-data')
        meta = {
            'zoo': np.float64(1.0),
            'foo': [
                np.ones(16),
                np.ones(16),
                np.ones(16)],
            'bar': {
                # Note: no support in JSON for complex numbers.
                'moo': np.uint64(42),
                'boo': np.full(8, 3, dtype=np.int32)
            }
        }
        mage = np.zeros([32, 32], dtype=np.float32)
        mrcName = str(tmp_path / 'testMage.mrc')
        mrcz.writeMRC(mage, mrcName, meta=meta)

        _, re_meta = mrcz.readMRC(mrcName)

        npt.assert_array_equal(re_meta['foo'][0], meta['foo'][0])
        npt.assert_array_equal(re_meta['bar']['boo'], meta['bar']['boo'])

    def test_enum_metadata(self, tmp_path):
        log.info('Testing Enum types in meta-data')

        class Axis(Enum):
            X = 0
            Y = 1

        meta = {'axes': [Axis.Y, Axis.X]}

        mage = np.zeros([4, 4], dtype=np.float32)
        mrcName = str(tmp_path / 'testMage.mrc')
        mrcz.writeMRC(mage, mrcName, meta=meta)

        _, re_meta = mrcz.readMRC(mrcName)

        assert 'Enum.Axis.X' in re_meta['axes']
        assert 'Enum.Axis.Y' in re_meta['axes']

    def test_MRC_append(self, rng, tmp_path):
        log.info('Testing appending to existing MRC stack, float-32')
        f32_stack = [rng.normal(size=(128, 96)).astype(np.float32) for _ in range(2)]

        mrcName = str(tmp_path / 'testStack.mrcs')

        for j, I in enumerate(f32_stack):
            mrcz.writeMRC(I, mrcName, pixelsize=[5.6, 3.4], compressor=None, idx=j)

        rereadMage, _ = mrcz.readMRC(mrcName, pixelunits=ANGSTROM)

        assert rereadMage.shape[0] == len(f32_stack)

        for testFrame, rereadFrame in zip(f32_stack, rereadMage):
            assert testFrame.dtype == rereadFrame.dtype
            npt.assert_array_almost_equal(testFrame, rereadFrame)

    def test_snake_case_aliases(self):
        # Same objects, not wrappers, so monkey-patching one affects the other.
        assert mrcz.read_mrc is mrcz.readMRC
        assert mrcz.write_mrc is mrcz.writeMRC
        assert mrcz.async_read_mrc is mrcz.asyncReadMRC
        assert mrcz.async_write_mrc is mrcz.asyncWriteMRC
        assert mrcz.read_mrc_header is mrcz.readMRCHeader
        assert mrcz.write_mrc_header is mrcz.writeMRCHeader
        assert mrcz.default_header is mrcz.defaultHeader
        assert mrcz.set_default_threads is mrcz.setDefaultThreads
        assert mrcz.read_dm4 is mrcz.readDM4
        assert mrcz.async_read_dm4 is mrcz.asyncReadDM4


#==============================================================================
# Regressions: one test per correctness bug fixed in 0.6.0.
#==============================================================================
class TestRegressions:

    def test_memmap_skips_the_header(self, rng, tmp_path):
        # `np.memmap` ignores the handle position; without `offset` this
        # returned the header reinterpreted as pixels.
        image = rng.normal(size=(5, 16, 12)).astype('float32')
        mrcName = str(tmp_path / 'memmap.mrc')
        mrcz.writeMRC(image, mrcName)

        mapped, _ = mrcz.readMRC(mrcName, useMemmap=True)
        npt.assert_array_equal(np.asarray(mapped), image)

    def test_memmap_skips_the_extended_header(self, rng, tmp_path):
        # An extended header pushes the data further in; `offset` must include it.
        image = rng.normal(size=(4, 16, 12)).astype('float32')
        mrcName = str(tmp_path / 'memmap_meta.mrc')
        mrcz.writeMRC(image, mrcName, meta={'padding': 'x' * 200})

        mapped, _ = mrcz.readMRC(mrcName, useMemmap=True)
        npt.assert_array_equal(np.asarray(mapped), image)

    def test_memmap_honors_idx(self, rng, tmp_path):
        image = rng.normal(size=(5, 16, 12)).astype('float32')
        mrcName = str(tmp_path / 'memmap_idx.mrc')
        mrcz.writeMRC(image, mrcName)

        mapped, _ = mrcz.readMRC(mrcName, useMemmap=True, idx=(1, 3))
        npt.assert_array_equal(np.asarray(mapped), image[1:4])

    def test_uint4_is_lossless_over_the_full_range(self, tmp_path):
        # The old decode floor(16a/15) is wrong only for nibble 15, and only in
        # the even-index branch. So all 16 values must appear at BOTH parities:
        # plain `arange(16)` leaves 15 at an odd index and passes even broken.
        row = np.arange(16, dtype='int8').repeat(2)
        image = np.broadcast_to(row, (2, 8, row.size)).copy()
        assert image[0, 0, 30] == 15, 'value 15 must sit at an even index'

        mrcName = str(tmp_path / 'uint4.mrc')
        mrcz.writeMRC(image, mrcName, dtype='uint4')

        reread, _ = mrcz.readMRC(mrcName)
        npt.assert_array_equal(reread, image)

    @requires_blosc
    def test_packed_bytes_is_a_scalar(self, rng, tmp_path):
        # The `[0]` was missing, so every header carried `(0,)`.
        image = rng.normal(size=(3, 32, 24)).astype('float32')
        mrcName = str(tmp_path / 'packed.mrcz')
        mrcz.writeMRC(image, mrcName, compressor='zstd', clevel=1, n_threads=1)

        _, header = mrcz.readMRC(mrcName)
        assert isinstance(header['packedBytes'], int)
        assert header['packedBytes'] == os.path.getsize(mrcName) - 1024

    @pytest.mark.parametrize('read_endian', ['le', 'be'])
    def test_big_endian_roundtrip(self, rng, tmp_path, read_endian):
        # A wrong endian guess used to raise IndexError. Passing "le" here
        # guesses wrong on purpose, so the detection has to recover.
        image = rng.normal(size=(3, 16, 12)).astype('float32')
        mrcName = str(tmp_path / 'bigendian.mrc')
        mrcz.writeMRC(image, mrcName, endian='be')

        reread, header = mrcz.readMRC(mrcName, endian=read_endian)
        npt.assert_array_almost_equal(reread, image)
        npt.assert_array_equal(header['dimensions'], image.shape)

    def test_big_endian_label_count(self, rng, tmp_path):
        # Written machine-native, so a big-endian reader saw 16777216 labels.
        mrcName = str(tmp_path / 'labels.mrc')
        mrcz.writeMRC(rng.normal(size=(2, 8, 8)).astype('float32'), mrcName, endian='be')

        with open(mrcName, 'rb') as f:
            f.seek(220)
            assert struct.unpack('>i', f.read(4))[0] == 1

    @pytest.mark.parametrize('wide,narrow', [('float64', 'float32'), ('complex128', 'complex64')])
    def test_write_does_not_mutate_caller_list(self, rng, tmp_path, wide, narrow):
        # In-place casting narrowed the caller's own arrays.
        frames = [make_stack(rng, wide, shape=(16, 12)) for _ in range(2)]
        originals = [frame.copy() for frame in frames]

        mrcz.writeMRC(frames, str(tmp_path / 'nomutate.mrc'))

        for frame, original in zip(frames, originals):
            assert frame.dtype == np.dtype(wide)
            npt.assert_array_equal(frame, original)

    @requires_blosc
    def test_lz4hc_has_its_own_type_code(self, rng, tmp_path):
        # lz4hc mapped to 2, so those files identified themselves as lz4.
        image = rng.normal(size=(2, 32, 24)).astype('float32')
        mrcName = str(tmp_path / 'lz4hc.mrcz')
        mrcz.writeMRC(image, mrcName, compressor='lz4hc', clevel=5, n_threads=1)

        with open(mrcName, 'rb') as f:
            f.seek(12)
            assert struct.unpack('<i', f.read(4))[0] == 3002

        reread, header = mrcz.readMRC(mrcName)
        assert header['compressor'] == 'lz4hc'
        npt.assert_array_almost_equal(reread, image)

    @pytest.mark.parametrize('pixelsize', [0.1, [0.1], (0.1,), np.array([0.1]), np.float64(0.1)])
    def test_singleton_pixelsize_does_not_overrun(self, rng, tmp_path, pixelsize):
        # A shape-(1,) pixelsize broadcast to (3, 3) and wrote nine floats over
        # the cell angles and axis associations behind cellsize.
        image = rng.normal(size=(5, 16, 12)).astype('float32')
        mrcName = str(tmp_path / 'cellsize.mrc')
        mrcz.writeMRC(image, mrcName, pixelsize=pixelsize)

        with open(mrcName, 'rb') as f:
            f.seek(40)
            cellsize = np.frombuffer(f.read(12), dtype='<f4')
            angles = np.frombuffer(f.read(12), dtype='<f4')
            axes = np.frombuffer(f.read(12), dtype='<i4')

        # dimensions are stored [nx, ny, nz], so cellsize is 0.1 * [12, 16, 5]
        npt.assert_array_almost_equal(cellsize, [1.2, 1.6, 0.5])
        npt.assert_array_equal(angles, [90.0, 90.0, 90.0])
        npt.assert_array_equal(axes, [1, 2, 3])

    def test_oversized_pixelsize_is_rejected(self, rng, tmp_path):
        with pytest.raises(ValueError, match='pixelsize must have 1, 2 or 3 elements'):
            mrcz.writeMRC(rng.normal(size=(2, 8, 8)).astype('float32'),
                          str(tmp_path / 'bad.mrc'), pixelsize=[1.0, 2.0, 3.0, 4.0])


#==============================================================================
# Compression backends: `blosc2` must read the blosc1 chunks we write, or
# upgrading would strand every MRCZ file already on disk.
#==============================================================================
@pytest.mark.skipif(not (mrcz.BLOSC_PRESENT and mrcz.BLOSC2_PRESENT),
                    reason='needs both `blosc` and `blosc2` installed')
class TestBloscBackendCompat:

    def test_written_chunks_are_blosc1_format(self, rng, tmp_path):
        # Chunk byte 0 is the format version. A 5 here would make the file
        # unreadable by every released version of this package.
        mrcName = str(tmp_path / 'version.mrcz')
        mrcz.writeMRC(rng.normal(size=(2, 32, 32)).astype('float32'), mrcName,
                      compressor='zstd', clevel=1, n_threads=1)

        with open(mrcName, 'rb') as f:
            f.seek(1024)
            assert f.read(1)[0] == 2

    @pytest.mark.parametrize('backend,expected_version', [(None, 2), ('blosc1', 2), ('blosc2', 5)])
    def test_backend_selects_the_chunk_format(self, rng, tmp_path, backend, expected_version):
        # blosc1 stays the default; blosc2 must be asked for explicitly.
        image = rng.normal(size=(3, 32, 24)).astype('float32')
        mrcName = str(tmp_path / f'backend_{backend}.mrcz')
        mrcz.writeMRC(image, mrcName, compressor='zstd', clevel=3, n_threads=1, backend=backend)

        with open(mrcName, 'rb') as f:
            f.seek(1024)
            assert f.read(1)[0] == expected_version

        reread, header = mrcz.readMRC(mrcName)
        npt.assert_array_almost_equal(reread, image)
        # The codec number is unchanged by the backend; only the container is.
        assert header['compressor'] == 'zstd'
        assert header['bloscFormat'] == (1 if expected_version == 2 else 2)
        assert header['backend'] == ('blosc1' if expected_version == 2 else 'blosc2')

    def test_blosc2_written_file_roundtrips_every_shared_codec(self, rng, tmp_path):
        image = rng.normal(size=(2, 32, 24)).astype('float32')
        for codec in ('blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd'):
            mrcName = str(tmp_path / f'{codec}_b2.mrcz')
            mrcz.writeMRC(image, mrcName, compressor=codec, clevel=3, n_threads=1,
                          backend='blosc2')
            reread, header = mrcz.readMRC(mrcName)
            assert header['compressor'] == codec, codec
            assert header['bloscFormat'] == 2, codec
            npt.assert_array_almost_equal(reread, image)

    @pytest.mark.parametrize('kwargs,exc,match', [
        ({'compressor': 'ndlz', 'backend': 'blosc1'}, ValueError, 'needs'),
        ({'compressor': 'zstd', 'backend': 'bogus'}, ValueError, 'backend must be'),
        ({'compressor': 'nosuchcodec'}, ValueError, 'Unknown compressor'),
    ])
    def test_invalid_backend_combinations_are_refused(self, rng, tmp_path, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            mrcz.writeMRC(rng.normal(size=(2, 8, 8)).astype('float32'),
                          str(tmp_path / 'bad.mrcz'), n_threads=1, **kwargs)

    def test_unknown_compressor_id_gives_a_real_error(self, rng, tmp_path):
        # Used to escape as a bare KeyError from inside the header parser.
        mrcName = str(tmp_path / 'future.mrcz')
        mrcz.writeMRC(rng.normal(size=(2, 8, 8)).astype('float32'), mrcName,
                      compressor='zstd', clevel=1, n_threads=1)
        with open(mrcName, 'r+b') as f:
            f.seek(12)
            f.write(struct.pack('<i', 999002))  # codec id 999, not in the enum

        with pytest.raises(ValueError, match='unrecognized MRCZ compressor id = 999'):
            mrcz.readMRC(mrcName)

    def test_blosc2_only_codec_reports_missing_plugin(self, rng, tmp_path):
        # A missing plugin must raise, not silently fall back to zstd and write
        # a file whose header names a codec the chunk does not hold.
        import blosc2

        # Probe with incompressible data: blosc2 run-length-encodes an all-zeros
        # buffer without ever calling the codec.
        probe = rng.normal(size=(64, 64)).astype('float32')
        try:
            blosc2.compress2(probe, codec=blosc2.Codec.GROK, clevel=1, typesize=4)
        except RuntimeError:
            pass
        else:
            pytest.skip('`blosc2-grok` is installed, so this codec does work here')

        with pytest.raises(RuntimeError, match='blosc2-grok'):
            mrcz.writeMRC(rng.normal(size=(2, 32, 24)).astype('float32'),
                          str(tmp_path / 'grok.mrcz'), compressor='grok', n_threads=1)

    def test_blosc2_reads_blosc1_chunks(self, rng, tmp_path):
        import blosc
        import blosc2

        from mrcz import ioMRC

        mrcName = str(tmp_path / 'compat.mrcz')
        testMage = rng.normal(size=(3, 32, 24)).astype('float32')
        mrcz.writeMRC(testMage, mrcName, compressor='zstd', clevel=1, n_threads=1)

        # Force each backend in turn and confirm they agree.
        assert blosc2.decompress(blosc.compress(testMage[0], 4, clevel=1)) == testMage[0].tobytes()

        original = ioMRC.BLOSC2_PRESENT
        try:
            ioMRC.BLOSC2_PRESENT = True
            via_blosc2, _ = mrcz.readMRC(mrcName)
            ioMRC.BLOSC2_PRESENT = False
            via_blosc1, _ = mrcz.readMRC(mrcName)
        finally:
            ioMRC.BLOSC2_PRESENT = original

        npt.assert_array_equal(via_blosc2, via_blosc1)
        npt.assert_array_almost_equal(via_blosc2, testMage)


#==============================================================================
# ioDM Test
#
# No real .dm4 fixture exists in the repo, so a minimal one is synthesized. It
# walks every parser branch: nested tag directories, auto-numbered anonymous
# directories, and array, singleton and struct payloads.
#==============================================================================
DM4_FLOAT32, DM4_INT32, DM4_UINT16 = 6, 3, 4


def _dm4_tag(name: bytes, infos: list[int], data: bytes) -> bytes:
    """A DM4 data tag, type 21."""
    field = b'%%%%' + struct.pack('>q', len(infos))
    field += b''.join(struct.pack('>q', i) for i in infos)
    field += data
    return b'\x15' + struct.pack('>h', len(name)) + name + struct.pack('>q', len(field)) + field


def _dm4_tagdir(name: bytes, children: list[bytes]) -> bytes:
    """A DM4 tag directory, type 20."""
    body = b'\x01\x01' + struct.pack('>q', len(children)) + b''.join(children)
    return b'\x14' + struct.pack('>h', len(name)) + name + struct.pack('>q', len(body)) + body


def _dm4_array(name: bytes, dtype_code: int, arr: np.ndarray) -> bytes:
    return _dm4_tag(name, [20, dtype_code, arr.size], arr.tobytes())


def _dm4_singleton(name: bytes, dtype_code: int, value, np_dtype: str) -> bytes:
    return _dm4_tag(name, [dtype_code], np.array(value, dtype=np_dtype).tobytes())


def _dm4_struct(name: bytes, fields: list[tuple]) -> bytes:
    """
    Structs sit inside the tag header rather than after it, so the field
    descriptors overlap the info array. `fields` is [(code, dtype, value), ...].
    """
    infos = [15, 0, len(fields)]
    for code, _, _ in fields:
        infos += [0, code]
    data = b''.join(np.array(v, dtype=d).tobytes() for _, d, v in fields)
    return _dm4_tag(name, infos, data)


def build_dm4(image: np.ndarray) -> bytes:
    ny, nx = image.shape

    image_data = _dm4_tagdir(b'ImageData', [
        _dm4_array(b'Data', DM4_FLOAT32, image.ravel()),
        _dm4_tagdir(b'Dimensions', [
            _dm4_singleton(b'', DM4_INT32, nx, '<i4'),
            _dm4_singleton(b'', DM4_INT32, ny, '<i4'),
        ]),
        _dm4_tagdir(b'Calibrations', [
            _dm4_tagdir(b'Brightness', [
                _dm4_singleton(b'Origin', DM4_FLOAT32, 0.25, '<f4'),
                _dm4_singleton(b'Scale', DM4_FLOAT32, 2.5, '<f4'),
                # ASCII stored one character per uint16, as Gatan does.
                _dm4_array(b'Units', DM4_UINT16, np.array([ord('e'), ord('-')], dtype='<u2')),
            ]),
        ]),
    ])

    image_tags = _dm4_tagdir(b'ImageTags', [
        _dm4_tagdir(b'Microscope Info', [
            _dm4_singleton(b'Voltage', DM4_FLOAT32, 300000.0, '<f4'),
            _dm4_singleton(b'Cs(mm)', DM4_FLOAT32, 2.7, '<f4'),
        ]),
        _dm4_tagdir(b'Acquisition', [
            _dm4_tagdir(b'Device', [
                _dm4_tagdir(b'Q', [
                    _dm4_struct(b'Pixel Size (um)',
                                [(DM4_INT32, '<i4', 5), (DM4_FLOAT32, '<f4', 14.0)]),
                    # The flip tags are only honored four levels below 'Device'.
                    _dm4_tagdir(b'R', [
                        _dm4_singleton(b'Horizontal Flip', DM4_INT32, 1, '<i4'),
                        _dm4_singleton(b'Vertical Flip', DM4_INT32, 0, '<i4'),
                    ]),
                ]),
            ]),
        ]),
    ])

    # ImageList holds one anonymous tag directory per image.
    image_list = _dm4_tagdir(b'ImageList', [_dm4_tagdir(b'', [image_data, image_tags])])

    header = struct.pack('>i', 4) + struct.pack('>q', len(image_list))
    header += struct.pack('>i', 1) + b'\x01\x01' + struct.pack('>q', 1)
    return header + image_list


class TestReadDM4:

    @pytest.fixture
    def dm4_file(self, rng, tmp_path):
        image = rng.normal(size=(6, 8)).astype('float32')
        path = tmp_path / 'synthetic.dm4'
        path.write_bytes(build_dm4(image))
        return str(path), image

    def test_image_data(self, dm4_file):
        path, image = dm4_file
        dm4 = mrcz.readDM4(path)

        assert len(dm4.im) == 1
        # readDM4 flips the fast axis to put the origin where MRC expects it.
        npt.assert_array_equal(dm4.im[0].imageData, image[:, ::-1])
        npt.assert_array_equal(dm4.im[0].shape, image.shape)

    def test_singleton_and_array_tags(self, dm4_file):
        path, _ = dm4_file
        info = mrcz.readDM4(path).im[0].imageInfo

        npt.assert_almost_equal(info['Voltage'], 300000.0)
        npt.assert_almost_equal(info['C3'], 2.7, decimal=6)
        npt.assert_almost_equal(info['IntensityOrigin'], 0.25)
        npt.assert_almost_equal(info['IntensityScale'], 2.5)
        assert info['IntensityUnits'] == 'e-'
        assert info['HorzFlip'] == 1
        assert info['VertFlip'] == 0

    def test_struct_tag(self, dm4_file):
        path, _ = dm4_file
        info = mrcz.readDM4(path).im[0].imageInfo

        assert list(info['DetectorPixelSize']) == [5, 14.0]

    def test_file_handle_is_released(self, dm4_file):
        path, _ = dm4_file
        dm4 = mrcz.readDM4(path)
        # Leaving it open raises ResourceWarning, which this suite errors on.
        assert dm4.f is None

    def test_async(self, dm4_file):
        path, image = dm4_file
        worker = mrcz.asyncReadDM4(path)
        npt.assert_array_equal(worker.result().im[0].imageData, image[:, ::-1])


def test(verbosity: int = 2) -> int:
    '''
    Run the ``pytest`` suite for the ``mrcz`` package.

    Retained for backward compatibility; `mrcz.test` is the same function.
    '''
    from mrcz import test as _test
    return _test(verbosity=verbosity)


# Stops pytest from collecting the runner shim as if it were a test case.
test.__test__ = False


if __name__ == '__main__':
    # Should generally call 'python -m pytest mrcz' for continuous integration
    raise SystemExit(test())
