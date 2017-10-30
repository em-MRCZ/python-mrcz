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

	   { 0:None, 1:'blosclz', 2:'lz4', 3:'lz4hc', 4:'snappy', 5:'zlib', 6:'zstd' }

   Unpacking is generally performed as follows::

	   mrcMode = numpy.mod(mrczMode, 1000)
	   compressor = numpy.floor_divide(mrczMode, 1000)

   In practice any **MODE** > 1000 indicates the use of a compression codec. 
   ``blosc`` will discover the actual codec used itself.

2. In the case where ``compressor != None``, starting at byte 1024 (or 1024 + 
   **EXTRA** if the extended header is used) a ``c-blosc`` header is found. The 
   ``c-blosc`` header format specification may be found here:

   https://github.com/Blosc/c-blosc/blob/master/README_HEADER.rst

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
4. **Word 36 (@ byte 14)**: Size of compressed data in bytes, including 
   ``blosc`` headers.  Present for convenience only.

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