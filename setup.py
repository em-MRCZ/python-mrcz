# -*- coding: utf-8 -*-
########################################################################
#
#       mrcz compressed MRC-file format package
#       License: MIT
#       Created: 02 November 2016
#       Author:  Robert A. McLeod - robbmcleod@gmail.com
#
########################################################################


from __future__ import print_function

import sys
from setuptools import setup



########### Check installed versions ##########
def exit_with_error(message):
    print('ERROR: %s' % message)
    sys.exit(1)

# Check for Python
if sys.version_info[0] == 2:
    if sys.version_info[1] < 7:
        exit_with_error("You need Python 2.7 or greater to install mrcz")
elif sys.version_info[0] == 3:
    if sys.version_info[1] < 3:
        exit_with_error("You need Python 3.3 or greater to install mrcz")
else:
    exit_with_error("You need Python 2.6/3.3 or greater to install mrcz")

tests_require = ['numpy']
########### End of checks ##########


# MRCZ version
major_ver = 0
minor_ver = 1
nano_ver = 4
branch = 'a2'

VERSION = "%d.%d.%d%s" % (major_ver, minor_ver, nano_ver, branch)
# Create the version.py file
open('mrcz/__version__.py', 'w').write('__version__ = "%s"\n' % VERSION)

# Global variables
classifiers = """\
Development Status :: 3 - Alpha
Intended Audience :: Developers
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Topic :: Software Development :: Libraries :: Python Modules
Topic :: System :: Archiving :: Compression
Operating System :: Microsoft :: Windows
Operating System :: Unix
"""

setup(name = "mrcz",
      version = VERSION,
      description = 'MRCZ meta-compressed image file-format library',
      long_description = """\

MRCZ is a highly optimized compressed version of the popular electron microscopy 
MRC image format.  It uses the Blosc meta-compressor library as a backend.  It 
can use a number of high-performance loseless compression codecs such as 'lz4' 
and 'zstd', it can apply bit-shuffling filters, and operates compression in a 
blocked and multi-threaded way to take advantage of modern multi-core CPUs.

""",
      classifiers = [c for c in classifiers.split("\n") if c],
      author = 'Robert A. McLeod',
      author_email = 'robbmcleod@gmail.com',
      url = 'http://github.com/em-MRCZ/python-mrcz',
      license = 'http://www.opensource.org/licenses/mit-license.php',
      platforms = ['any'],

      tests_require=tests_require,
      packages = ['mrcz'],

)
