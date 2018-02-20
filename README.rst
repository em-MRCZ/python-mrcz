==============================================
Python MRCZ meta-compressed file-format module
==============================================

``mrcz`` is a package designed to supplement the venerable MRC image file 
format with a highly efficient compressed variant, using the ``blosc`` 
meta-compressor library to shrink files on disk and greatly accelerate file 
input/output for the era of "Big Data" in electron and optical microscopy.

Python versions 2.7, 3.4-3.6 are supported.

``mrcz`` is currently considered to be a `beta` development state.

``mrcz`` is released under the BSD 3-clause license.

Installation
------------

A scientific Python installation (such as Anaconda, WinPython, or Canopy) is 
advised. After installation of your Python environment, from a command prompt 
type::

    pip install mrcz

``mrcz`` has the following dependencies:

* ``numpy``
* ``blosc`` (optionally, but highly recommended)

Feature List
------------

* Import: DM4, MRC, MRCZ formats
* Export: MRC, MRCZ formats
* Compress and bit-shuffle image stacks and volumes with ``blosc`` meta-compressor
* Asynchronous read and write operations.
* Support in the ``hyperspy`` electron microscopy package.

Documentation
-------------

Documentation is hosted at http://python-mrcz.readthedocs.io/

Authors
-------

See ``AUTHORS.txt``.

Citations
---------

* R.A. McLeod, R. Diogo-Righetto, A. Stewart, H. Stahlberg, "MRCZ – A file 
  format for cryo-TEM data with fast compression," Journal of Structural Biology,
  201 (3) (March 2018): 252-267, https://doi.org/10.1016/j.jsb.2017.11.012
* A. Cheng et al., "MRC2014: Extensions to the MRC format header for electron 
  cryo-microscopy and tomography", Journal of Structural Biology 192(2): 146-150, 
  November 2015, http://dx.doi.org/10.1016/j.jsb.2015.04.002
* V. Haenel, "Bloscpack: a compressed lightweight serialization format for 
  numerical data", arXiv:1404.6383


