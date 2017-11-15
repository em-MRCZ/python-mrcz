0.3.1
-----

Added ascii identifier label 'MRCZ' + <__version__> to the labels.  I.e. at 
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


