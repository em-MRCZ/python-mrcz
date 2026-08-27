0.6.0
-----
Compatibility release. ``mrcz`` was non-functional on NumPy 2.x: every MRC read
raised ``TypeError`` and ``readDM4`` could not parse anything. 19 test errors to
zero.

Compatibility
~~~~~~~~~~~~~

* NumPy 2.x. ``int(np.fromfile(..., count=1))`` on the header path raised
  ``TypeError``, breaking every ``readMRC``. ``ndarray.tostring()`` (removed in
  2.0) broke ``readDM4`` in five places. ``np.arange()`` on a shape-(1,) array
  broke the DM4 struct-tag parser. Header scalars are now written via
  ``np.array(value, dtype=...)`` and the MRC mode computed as a Python ``int``,
  so NEP 50 promotion cannot change it.
* Python 3.13+. ``unittest.makeSuite`` was removed, so ``mrcz.test()`` was
  already broken. Tests are now ``pytest``; ``mrcz.test()`` is a thin
  ``pytest.main()`` shim importing ``pytest`` lazily.
* Fixed the invalid escape sequences (``'\AA'``, ``'\mum'``) rather than
  suppressing their ``SyntaxWarning``, and dropped the ``warnings.filterwarnings``
  call 0.5.8 added to ``__init__.py``. Both spellings give the same string at
  runtime, so callers are unaffected.
* Removed the Python 2 scaffolding: ``__future__`` imports, the
  ``concurrent.futures`` backport fallbacks (one referenced an un-imported
  ``sys``), ``u''`` prefixes, ``class Foo(object)``. ``logger.warn`` ->
  ``logger.warning``, ``== None`` -> ``is None``, and bare ``except:`` clauses
  now name their exceptions.
* Added type hints to all public signatures, and snake_case aliases
  (``read_mrc``, ``write_mrc``, ``read_dm4``, ...) bound to the same objects, so
  the camelCase names and downstream consumers such as ``hyperspy`` are
  unaffected.

Compression backends
~~~~~~~~~~~~~~~~~~~~

* ``blosc2`` is now preferred for reading, as it decompresses both the blosc1
  (version-2) and blosc2 (version-5) chunk formats.
* ``writeMRC`` gained ``backend``: ``None`` (default), ``'blosc1'`` or
  ``'blosc2'``. The default resolves to ``'blosc1'`` whenever it can emit the
  requested codec, so output is unchanged unless asked. ``'blosc2'`` writes
  version-5 chunks, which ``mrcz < 0.6`` cannot read.
* No new header flag was needed: chunk byte 0 is already the format version, and
  the MRCZ spec already treats the **MODE** codec number as advisory. Reads
  surface ``header['bloscFormat']`` (1 or 2) and ``header['backend']``.
* The codec enumeration gained the blosc2-only codecs, 7-15: ``ndlz``,
  ``zfp_acc``, ``zfp_prec``, ``zfp_rate``, ``openhtj2k``, ``grok``, ``openzl``,
  ``j2k``, ``htj2k``. Requesting one forces ``backend='blosc2'``. Two caveats:

  - Most ship as separate plugin distributions and raise if absent. The blosc2
    write path uses ``blosc2.compress2``, not the ``blosc2.compress`` shim, which
    *silently falls back to zstd* and would write a file whose header names a
    codec the chunk does not hold.
  - The ``zfp_*`` family needs blosc2 NDArray metadata that per-frame MRCZ chunks
    lack, so it barely compresses. A warning is logged. Prefer ``zstd``.

* ``n_threads`` is now actually applied when decompressing; the old code assigned
  ``blosc.nthreads``, which only rebinds a module global.
* Removed the ``c-mrcz`` cross-compatibility tests. That project is long
  undeveloped and the executable is absent everywhere, so the 15 tests had been
  silently skipping. Written files are unchanged.

Correctness fixes
~~~~~~~~~~~~~~~~~

Long-standing bugs unrelated to NumPy 2.x, each shipped broken for years because
no test covered the path. Each now has a regression test confirmed to fail
against the old code.

* ``useMemmap=True`` returned the file header as image data. ``np.memmap`` maps
  from byte 0 regardless of handle position and needs an explicit ``offset``. A
  ``float32`` stack came back as ``[1.7e-44, 2.2e-44, ...]``.
* ``dtype='uint4'`` did not round-trip. The low nibble decoded as
  ``left_shift(x, 4) / 15``, i.e. ``floor(16a/15)``. Truncation hid it for every
  value but 15, which decoded to 16. Now ``x & 0x0F``.
* Big-endian files raised ``IndexError``: a wrong endianness guess indexed ``[0]``
  into a 0-d array, and would have byteswapped an already-misdecoded value. The
  bytes are now reinterpreted with ``.view()``.
* ``header['packedBytes']`` was the tuple ``(0,)``; the ``[0]`` was missing, and
  it decoded machine-native rather than at the file's byte order.
* ``writeMRC`` mutated the caller's list, narrowing their ``float64`` or
  ``complex128`` frames in place during what should be a read-only operation.
* A one-element ``pixelsize`` corrupted three header fields: a shape-(1,) array
  broadcast to (3, 3) and wrote nine floats over the cellsize field, the cell
  angles, and the MAPC/MAPR/MAPS axis associations. More than three elements is
  now a clear ``ValueError``.
* The label count ignored the requested byte order, so big-endian files claimed
  16777216 labels.
* ``lz4hc`` was written with the ``lz4`` type code, contradicting
  ``COMPRESSOR_ENUM``. This is the only change that alters bytes on disk: new
  ``lz4hc`` files use mode 3002, not 2002. Existing files are unaffected, since
  blosc records the codec in its own chunk header.
* An unrecognized codec number escaped as a bare ``KeyError`` from the header
  parser; now a ``ValueError`` naming the id.
* ``readDM4`` now opens its file in a ``with`` block, so the handle is released
  when parsing raises. Its ``verbose`` output goes to the ``'MRCZ'`` logger
  instead of ``print``.

Validation
~~~~~~~~~~

* A 17-file corpus covering every dtype, pixel unit, compressor, list layout and
  JSON extended header was written before and after and compared by SHA-256: all
  byte-identical, ``lz4hc`` aside.
* 21 real files written by 0.3.6 through 0.5.6 (uncompressed and zstd; uint16,
  uint32, float32, complex64; monolithic, list-of-2D and multi-slice list-of-3D)
  read identically under 0.5.10 and this release, pixels and every shared header
  key. Files written with the default backend remain readable by 0.5.10; those
  written with ``backend='blosc2'`` fail there with a blosc error rather than
  misreading.
* ``readDM4`` gained coverage for the first time, via a synthetic DM4 fixture
  exercising nested tag directories, anonymous directory auto-numbering, and the
  array, singleton and struct payloads.

Packaging
~~~~~~~~~

* Added ``compression``, ``blosc2``, ``faster-json`` and ``dev`` extras, trove
  classifiers, project URLs, and a ``pytest`` config treating new deprecation
  warnings as failures.
* ``MANIFEST.in`` referenced ``RELEASE_NOTES.txt`` but the file is
  ``RELEASE_NOTES.rst``, so release notes were missing from the sdist.

0.5.10
------
- Remove use of `distutils` as it is deprecated. Using `packaging` instead.

0.5.9
-----
* Did not properly specify how to find source files so 0.5.8 wheel was broken.
  (In my defense I've used `src\<package>` builds for almost five years now.)

0.5.8
-----
* Suppress warnings about SyntaxError: invalid escape sequences.
* Switch to using `pyproject.toml` over `setup.py`.

0.5.7
-----
* Renamed `np.product` to `np.prod` as the old name is deprecated in NumPy 2.0.

0.5.6
-----
* In 0.5.5 a for-loop was omitted which lead to every frame being the zeroth 
  frame in the stack. For this reason, upgrading from 0.5.5 is _strongly_ 
  recommended.

0.5.5
-----
* Meta-data with keys that match those used in the header could accidently 
  overwrite critical values, such as 'dimensions'. Any keys in the JSON 
  meta-dictionary that overlap with the standard values are now ignored.
* Integration tests are now performed for Python 3.8.

0.5.4
-----
* Added support for serialization of Python Enum objects in JSON serialization 
  of meta-data.

0.5.3
-----
* Ricardo Righetto added the means to append frames to a stack.
* Support for Python 3.4 was dropped as it is past end-of-life by Python.org.

0.5.2
-----
* Improved on serialization of non-standard (i.e. NumPy) types in JSON-ized 
  meta-data by making use of the `default` callable in `json.dumps`. In particular
  deeply nested NumPy types should now serialize without erroring. Note that 
  there is no support for complex numbers in JSON meta-data, as JSON itself 
  does not support it by default.

0.5.1
-----
* Versions of MRCZ <= 0.4.1 were improperly writing the dimensions into the 
  (Mx, My, Mz) volume fields. Added a check for the MRCZ version tag, and if 
  an older file is found, it defaults to ``slices == 1``, i.e. one 2D frame 
  per element in the returned list.
  - In order to suppress the warning message, files can be read into memory and 
    re-saved. A utility script for batch processing is provided in 
    ``utils\update_mrcz_0.5.0.py``.

0.5.0
-----
* Added support for lists of 3D `numpy.ndarray` objects. This is largely intended 
  to support multi-channel STEM time series. Stores the number of channels per 
  frame in the `MZ` value of the MRC2014 header, which must be uniform for 
  every ndarray in the list. Any MRCZ archive that has a 'strides' key in the 
  JSON metadata will be returned as a list of arrays. 
  - See http://www.ccpem.ac.uk/mrc_format/mrc2014.php for header details
  - `asList` keyword arguments have been removed.
* Fixed a bug in casting from float64/complex128 that was not actually casting.
* Cleaned up the code to be more PEP8 compliant.

0.4.1
-----
* Improved docstrings in `ioDM4.py`.
* Added `asyncReadDM4` function, analogous to `asyncReadMRC`.

0.4.0
-----
* Fix a minor bug with casting for lists of arrays
* Improved uncompressed write times by not using list comphrension
* Add scaling block size for small format images (e.g. Medipix) to scale to 
  the number of threads.
* If the passed arrays are C_CONTIGUOUS and ALIGNED, `writeMRC` will use 
  `blosc.compress_ptr` instead of coverting the array to a `bytes` object 
  which is a significant speedup.

0.3.8
-----
* Auto-casts `np.float64` -> `np.float32` and `np.complex128` -> `np.complex64` 
  but logs a warning to the user.

0.3.7
-----
* Updated MANIFEST.in and `setup.py` to make Conda-forge happy.

0.3.6
-----

* `mrcz.ReliablePy` must be imported explicitely now, as it has requirements 
  that the base `mrcz` package does not. This file may be removed in the 
  future if no users are using it.

0.3.5
-----

* If ``blosc`` is not installed and the user attempts to operate with compression 
  on an ``ImportError`` is raised.
* Documentation now using Numpy docstrings.

0.3.4
-----

* Add (temporarily) MRC types for `uint32` and `int32` to support 24-bit detectors.
  May break in the future, as the CCP-EM committee should make the final decision
  on such enumerations.
* Added handling of NumPy scalars (i.e. `np.float32(1.0)`) in metadata so that 
  JSON serialization does not generate errors. Values will be case to Python 
  `int` or `float` as appropriate.

0.3.3
-----

* Removed use of star-expansion of args as it breaks Python 2.7/3.4.

0.3.2
-----

* Made `blosc` an optional dependency due to difficulties involved in building
  wheels for PyPi.
* Implemented reading/writing of `list` of equally-shaped 2D `ndarray`s instead of 
  a single 3D `ndarray`, where the `list` represents the Z-axis. This approach 
  can be helpful for larger arrays that do not have to be continuous as the 
  operating system can more easily interleave them into memory.

0.3.1
-----

* Added ascii identifier label 'MRCZ' + <__version__> to the labels.  I.e. at 
  byte 224 in the header will appear b'MRCZ0.3.1'

0.3.0
-----

* Documentation now available at http://python-mrcz.readthedocs.io/
* Added continuous integration testing with Appveyor and TravisCI, which was 
  previously handled by `c-mrcz`.
* Added handling for `dask.array.core.Array` objects.
* `numpy.ndarrays` inside `meta` dictionaries will be converted to `list` 
  objects to facilitate serialization.
* Updated license to BSD-3-clause from BSD-2-clause.
* Various bug fixes.

0.2.1-4
-------

* Various bug fixes to incorporate into Hyperspy.

0.2.0
-----

* Added support for asynchronous reading and writing.

0.1.4a1
-------

* Fixed a bug with the machine-stamp not being converted to bytes properly.

0.1.4a0
-------

* Fixed a bug in import of mrcz from ReliablePy

0.1.3a2
-------

* Added ReliablePy, an interface for Relion .star and Frealign .par files.
* Fixes to maintain cross-compatibility with `c-mrcz`.  Main functions are
  readMRC and writeMRC.  readMRC always returns a header now.
* Added mrcz_test suite, which also tests `c-mrcz` if it's found in the path.
* Fixed bugs related to `mrcz_test.py`


0.1.1a1
-------

* Renamed 'cLevel' to 'clevel' to maintain consistency with `blosc` naming 
  convention.
* Updated license from MIT to BSD 2-clause.

0.1.0dev0
---------

Initial commit


