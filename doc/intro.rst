Introduction
============

MRCZ is a union of the MRC file format with ``blosc`` meta-compression.  ``blosc``
is not a compression algorithm, rather it is a standard that supports most 
popular compression algorithms. It can also apply lossless filters that 
improve compression performance, such as the ``bitshuffle`` filter. It achieves 
high-performance through the use of multi-threading the supported compression
codecs. Generally you should expect MRCZ to result in faster file read/write 
rates, as the compression is faster than hard drive read/write rates, as well
as near entropy-limited compression ratios.  So you get something for nothing.

Typical usage patterns are::

    imageData, imageMeta = mrcz.readMRC('my_filename.mrcz')

where ``imageData`` is a ``numpy.ndarray`` and ``imageMeta`` is a Python ``dict`` 
containing metadata.  After some manipulation, you may want to then save to disk
so the file can be passed into a third-party application, such as a CTF estimation 
tool. Here for maximum compatibility we will save it uncompressed (which 
is the default keyword argument for ``compressor``)::

    mrcz.writeMRC( imageData, 'passed_file.mrc', compressor=None )

Alternatively you may want to save an archival compression version of your 
data in the background using the asynchronous feature.  In this case, the exact 
time when the write finishes is typically not a concern (although see the the 
function documentation for finer control)::

    mrcz.asyncWriteMRC( imageData, 'my_newfile.mrcz', meta=newMeta, compressor='zstd', clevel=1 )

See the API reference docs for detailed information on usage. The recommended 
compression codecs and levels are:

 * ``compressor='zstd'`` and ``clevel=1`` for general archival use.
 * ``compressor='lz4'`` and ``clevel=9`` for speed-critical applications.

The ``bitshuffle`` filter is always used in MRCZ compressed files as it was 
found to improve both compression rate and ratio with representative 
electron microscopy data.