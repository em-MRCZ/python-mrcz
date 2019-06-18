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


