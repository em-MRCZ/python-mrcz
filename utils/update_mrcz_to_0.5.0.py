import glob, sys, os, os.path as path
import numpy as np
import mrcz
from distutils.version import StrictVersion
import shutil

args = sys.argv
directory = '.'
"""
This script is used for updating a directory of older MRCZ files from versions
<= 0.4.1 to the 0.5.0 standard, so that the volume sampling is set to the 
default (Mx, My, Mz) = [0, 0, 1].
"""

if len(sys.argv) > 1:
    directory = sys.argv[1]

mrcz_files = glob.glob(path.join(directory, '*.mrc'), recursive=False)
mrcz_files.extend(glob.glob(path.join(directory, '*.mrcz'), recursive=False))

backup_dir = path.join(directory, 'backup')
os.makedirs(backup_dir)
for filename in mrcz_files:
    shutil.copy(filename, path.join(backup_dir, path.basename(filename)))

    data, meta = mrcz.readMRC(filename)

    compressor = meta['compressor'] if 'compressor' in meta else None
    clevel = meta['clevel'] if 'clevel' in meta else 1
    voltage = meta['voltage'] if 'voltage' in meta else 0.0
    C3 = meta['C3'] if 'C3' in meta else 0.0
    gain = meta['gain'] if 'gain' in meta else 1.0
    pixelsize = meta['pixelsize'] if 'pixelsize' in meta else [0.1,0.1,0.1]
    pixelunits = meta['pixelunits'] if 'pixelunits' in meta else u'\\AA'
    
    if 'packedBytes' in meta:
        meta.pop('packedBytes')
    if 'minImage' in meta:
        meta.pop('minImage')
    if 'maxImage' in meta:
        meta.pop('maxImage')
    if 'meanImage' in meta:
        meta.pop('meanImage')
    if 'extendedBytes' in meta:
        meta.pop('extendedBytes')

    mrcz.writeMRC(data, filename, meta=meta,
        pixelsize=pixelsize, pixelunits=pixelunits,
        voltage=voltage, C3=C3, gain=gain,
        compressor=compressor, clevel=clevel)
    
    # Try and read every file back in.
    try:
        mrcz.readMRC(filename)
    except:
        raise ValueError('{} could not be re-read'.format(filename))
    
print('__All__ originals were backed up to %s, remember to delete them' % path.join(directory, 'backup'))

    