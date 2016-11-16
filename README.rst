============================================================
MRCZ meta-compressed file-format package (Python 2.7/3.3-3.5)
============================================================

Author: Robert A. McLeod

Email: robbmcleod@gmail.com

python-MRCZ is a package designed to supplement the venerable MRC image file format with a highly efficient compressed variant, using the Blosc meta-compressor library to shrink files on disk and greatly accelerate file input/output for the era of "Big Data" in electron (and optical) microscopy.

python-MRCZ is currently in alpha. 

python-MRCZ is released under the BSD license.

Installation
------------

A scientific Python installation (such as Anaconda, WinPython, or Canopy) is advised.  After installation, from a command prompt type::

    pip install mrcz

python-MRCZ has the following dependencies:

* `numpy`
* `blosc`

Feature List
------------

* Import: DM4, MRC,
* Compress and bit-shuffle image stacks and volumes with `blosc` meta-compressor


Citations
---------

* A. Cheng et al., "MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography", Journal of Structural Biology 192(2): 146-150, November 2015, http://dx.doi.org/10.1016/j.jsb.2015.04.002
* V. Haenel, "Bloscpack: a compressed lightweight serialization format for numerical data", arXiv:1404.6383


