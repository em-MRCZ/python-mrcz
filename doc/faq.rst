Frequently Asked Questions
==========================

**Can I access individual slices in a compressed MRCZ file?**

Currently not, as ``blosc`` does not record the indices of its individual 
compressed blocks. The new feature of 'super-chunks', equivalent to slices or 
frames in the context of microscopy, is expected to be implemented in 
``c-blosc2``, currently under development at https://github.com/Blosc/c-blosc2

