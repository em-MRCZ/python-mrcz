MRCZ Specification
==================

In general MRCZ follows the CCPEM MRC2014 standard as outlined here:

http://www.ccpem.ac.uk/mrc_format/mrc2014.php

Please note we count bytes starting from ``0``. CCP-EM counts bytes starting 
from ``1``.

Required deviations from CCPEM MRC2014 standard
-----------------------------------------------

1. **Word 4 (@ byte 16)**: The **MODE** parameter is now the sum of the MRC2014 
   **MODE** plus the ``blosc`` compression used * 1000.

   The compressor enumeration is::

	   { 0:None, 1:'blosclz', 2:'lz4', 3:'lz4hc', 4:'snappy', 5:'zlib', 6:'zstd',
	     7:'ndlz', 8:'zfp_acc', 9:'zfp_prec', 10:'zfp_rate',
	     11:'openhtj2k', 12:'grok', 13:'openzl', 14:'j2k', 15:'htj2k' }

   Numbers 0-6 are what ``c-blosc1`` can emit. Numbers 7 and above exist only
   in ``c-blosc2``, so a file using one carries version-5 chunks (see item 2)
   and needs ``blosc2``, in several cases plus a separate plugin package.

   *Note*: before 0.6.0 this library wrote ``'lz4hc'`` as 2 rather than 3, so
   files from ``mrcz <= 0.5.10`` identify lz4hc data as ``'lz4'``. Pixel data is
   unaffected, since the codec actually used is recorded in the blosc chunk.

   Unpacking is generally performed as follows::

	   mrcMode = numpy.mod(mrczMode, 1000)
	   compressor = numpy.floor_divide(mrczMode, 1000)

   In practice any **MODE** > 1000 indicates the use of a compression codec. 
   ``blosc`` will discover the actual codec used itself.

2. In the case where ``compressor != None``, starting at byte 1024 (or 1024 + 
   **EXTRA** if the extended header is used) a ``c-blosc`` header is found. The 
   ``c-blosc`` header format specification may be found here:

   https://github.com/Blosc/c-blosc/blob/master/README_HEADER.rst

   Byte 0 of that header is the chunk format version, and is the authoritative
   record of which library wrote the data section::

	   2 -> c-blosc1 chunks, readable by `blosc` and `blosc2`
	   5 -> c-blosc2 chunks, readable only by `blosc2`

   **MODE** names the codec but not the container, so a reader should consult
   this byte to decide whether it can decode the file. ``mrcz`` writes
   version-2 by default and reports ``header['bloscFormat']`` (``1`` or ``2``)
   on read. Version-5 is written only for ``writeMRC(backend='blosc2')``, a
   blosc2-only codec, or when ``blosc`` is absent.

   ``blosc`` is limited to ``2**31`` bytes per chunk. Chunking for compression is 
   accomplished by compressing each slice/frame in the z-axis with a separate 
   call to ``blosc.compress()``.  Therefore the data section consists of **NZ** 
   structs of ``c-blosc`` headers followed by the packed bytes for the associated 
   slice/frame.  


Optional deviations from CCPEM MRC2014 standard
-----------------------------------------------

1. **Word 33 (@ byte 132)**: Accelerating voltage in keV, float-32 format. 
   **Deprecated**.
2. **Word 34 (@ byte 136)**: Spherical aberration in mm, float-32 format. 
   **Deprecated**.
3. **Word 35 (@ byte 140)**: Detector gain in e^-/DN, defaults to 1.0. 
   **Deprecated**.
4. **Word 36-37 (@ byte 14)**: Size of compressed data in bytes stored as a 64-bit
   integer, including ``blosc`` headers. Present for convenience only.
5. **Word 57 (@ byte 224)**: The ascii-encoded identifier label 'MRCZ<version>'.
   For example, ``b'MRCZ0.3.1'.

Failure to include any of these variables will not result in an exception.

JSON extended meta-data
^^^^^^^^^^^^^^^^^^^^^^^

When the keyword argument ``meta`` is used with ``writeMRC`` and 
``asyncWriteMRC`` the passed dictionary will be converted to UTF-8 encoded JSON 
and written into the extended header. This is indicated by the ascii-encoded 
bytes ``'json'`` written into the **EXTTYP** variable of the MRC2014 header. The 
length of the encoded JSON metadata is stored in the **EXTRA** variable of the 
MRC2014 header.

*Note*: ``python-rapidjson`` is preferred but the standard library ``json`` 
module is used as a fallback.