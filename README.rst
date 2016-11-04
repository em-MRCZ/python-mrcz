=====
MRCZ Compressed MRC file-format package
=====

Author: Robert A. McLeod

Email: robbmcleod@gmail.com

MRCZ is a package designed to supplement the venerable MRC image file format with a highly efficient compressed variant, using the Blosc meta-compressor library 
to shrink files on disk and greatly accelerate file input/output for the era of "Big Data" in electron (and optical) microscopy.

`mrcz` is currently in alpha. 

For help on installation please see the wiki page: https://github.com/em-MRCZ/python-mrcz/wiki

MECZ has the following dependencies:

* `numpy`
* `blosc`

MRCZ is MIT license.

Feature List
-----

* Import: DM4, MRC,
* Compress images with `blosc` meta-compressor


Citations
-----

* A. Cheng et al., "MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography", Journal of Structural Biology 192(2): 146-150, November 2015, http://dx.doi.org/10.1016/j.jsb.2015.04.002
* V. Haenel, "Bloscpack: a compressed lightweight serialization format for numerical data", arXiv:1404.6383


