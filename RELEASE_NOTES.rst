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


