# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:44:09 2016

@author: Robert A. McLeod
"""


import mrcz
import numpy as np
import numpy.testing as npt
import os, os.path, sys
import subprocess
import tempfile
import unittest
try:
    from termcolor import colored
except:
    def colored( string ):
        return string

float_dtype = 'float32'
fftw_dtype = 'complex64'
tmpDir = tempfile.gettempdir()

#==============================================================================
# ioMRC Test
# 
# Build a random image and save and re-load it.
#==============================================================================
class test_ioMRC(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_vanillaMRC(self):
        testMage = np.random.normal( size=[2,128,136] ).astype( float_dtype )
        mrcName = os.path.join( tmpDir, "testMage.mrc" )
        
        pixelsize = 1.326
        
        mrcz.ioMRC.MRCExport( testMage, mrcName,
                                pixelsize=pixelsize, pixelunits=u"\AA",
                                voltage=300.0, C3=2.7, gain=1.05 )
        rereadMage, rereadHeader = zorro.ioMRC.MRCImport( mrcName, pixelunits=u"\AA",
                                                         returnHeader=True )
        os.remove( mrcName )
        
        npt.assert_array_almost_equal( testMage, rereadMage )
        npt.assert_array_equal( rereadHeader['voltage'], 300.0 )
        npt.assert_approx_equal( rereadHeader['pixelsize'][0], pixelsize )
        npt.assert_array_equal( rereadHeader['pixelunits'], u"\AA" )
        npt.assert_array_equal( rereadHeader['C3'], 2.7 )
        npt.assert_array_equal( rereadHeader['gain'], 1.05 )
        
        
        
    def test_4bitMRC(self):
        testMage = np.random.uniform( high=10, size=[2,128,136] ).astype( 'uint8' )
        mrcName = os.path.join( tmpDir, "testMage.mrc" )
                
        pixelsize = [1.2, 5.6]
        
        mrcz.ioMRC.MRCExport( testMage, mrcName, dtype='uint4',
                                pixelsize=pixelsize, pixelunits=u"\AA",
                                voltage=300.0, C3=2.7, gain=1.05 )
                                
        rereadMage, rereadHeader = zorro.ioMRC.MRCImport( mrcName, pixelunits=u"\AA",
                                                         returnHeader=True )                     
        os.remove( mrcName )
        
        assert( np.all(testMage.shape == rereadMage.shape) )
        assert( testMage.dtype == rereadMage.dtype )
        npt.assert_array_almost_equal( testMage, rereadMage )
        npt.assert_array_equal( rereadHeader['voltage'], 300.0 )
        npt.assert_array_almost_equal( rereadHeader['pixelsize'][:2], pixelsize )
        npt.assert_array_equal( rereadHeader['pixelunits'], u"\AA" )
        npt.assert_array_equal( rereadHeader['C3'], 2.7 )
        npt.assert_array_equal( rereadHeader['gain'], 1.05 )
        pass
        
        
    def test_MRCZ(self):
        testMage = np.random.uniform( high=10, size=[2,128,136] ).astype( 'int8' )
        mrcName = os.path.join( tmpDir, "testMage.mrcz" )
                
        pixelsize = [1.2, 5.6, 3.4]
        
        mrcz.ioMRC.MRCExport( testMage, mrcName,
                                pixelsize=pixelsize, pixelunits=u"\AA",
                                voltage=300.0, C3=2.7, gain=1.05,
                                compressor='zstd', cLevel=1, n_threads=4 )
                                
        rereadMage, rereadHeader = zorro.ioMRC.MRCImport( mrcName, pixelunits=u"\AA",
                                                         returnHeader=True )
        os.remove( mrcName )
        
        assert( np.all(testMage.shape == rereadMage.shape) )
        assert( testMage.dtype == rereadMage.dtype )
        npt.assert_array_almost_equal( testMage, rereadMage )
        npt.assert_array_equal( rereadHeader['voltage'], 300.0 )
        npt.assert_array_almost_equal( rereadHeader['pixelsize'], pixelsize )
        npt.assert_array_equal( rereadHeader['pixelunits'], u"\AA" )
        npt.assert_array_equal( rereadHeader['C3'], 2.7 )
        npt.assert_array_equal( rereadHeader['gain'], 1.05 )
        pass
    

    
if __name__ == "__main__":
    from mrcz import __version__
    print( "MRCZ TESTING FOR VERSION %s " % __version__ )
    unittest.main( exit=False )
    

    
    



                     
