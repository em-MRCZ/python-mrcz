# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:34:02 2016

@author: rmcleod
"""
import blosc
import numpy as np
import time
import mrcz
import tempfile, os, os.path, glob

print( "==== Generating simulated Poisson stacks ====" )
print( "Note: will write files to: %s. (they will not be deleted as they can be re-used!)" % tempfile.gettempdir() )


stackShape = [50,4096,4096]
doseLevelList = [ 0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0 ]
doseFilenames = []
poissonDataArray = np.zeros( [len(doseLevelList), stackShape[0],stackShape[1],stackShape[2]]  )
for I, dose in enumerate(doseLevelList):
    poissonName = os.path.join(tempfile.gettempdir(), "poisson%f.mrcz" % dose )
    doseFilenames.append( poissonName )
 
    if not os.path.isfile( poissonName ):
        print( "Generating Poisson random numbers and writing %s" % poissonName )
        poissonDataArray[I,...] = np.random.poisson( lam=dose, size=stackShape ).astype('int8')
        mrcz.MRCExport( poissonDataArray[I,...], poissonName, compressor='zstd' )

    else:
        # Load from disk
        print( "Loading from disk: %s" % poissonName )
        poissonDataArray[I,...] = mrcz.MRCImport( poissonName )
        

print( "==== Starting compression benchmark ====" )

def doCompression( dataStack, 
                  compressor='zstd', blocksize=2**20, n_threads=16, 
                  shuffle=blosc.BITSHUFFLE, clevel=5 ):

    blosc.set_blocksize( blocksize )
    blosc.set_nthreads( n_threads )
    typeSize = dataStack.dtype.itemsize
    packedDataList = [None] * dataStack.shape[0]
    for J in np.arange(dataStack.shape[0]):
        packedDataList[J] = blosc.compress( dataStack[J,:,:], typesize=typeSize,
                                     clevel=clevel, shuffle=shuffle, cname=compressor )
        
    return packedDataList

def doDecompression( packedDataList, shape, n_threads ):
    blosc.set_nthreads( n_threads )
    dataList = [None] * len(packedDataList)
    for J in np.arange(len(packedDataList) ):
#        dataStack[J,:,:] = np.reshape( 
#            np.frombuffer( blosc.decompress( packedDataList[J] ), dtype='uint8' ),
#            shape[1:] )
        # Something here Numpy-side is very slow, so let's not include that in our 
        # benchmark.
        dataList[J] = blosc.decompress( packedDataList[J] )
    return dataList




#t_half0 = time.time()
#halfimage = dm4image_8bit[:,:,::2] + np.left_shift(dm4image_8bit[:,:,1::2],4)
#t_half1 = time.time()
#restoreimage = np.empty( header['dimensions'], dtype='uint8' )
##image[0::2] = np.left_shift(interlaced_image,4)/16
##image[1::2] = np.right_shift(interlaced_image,4)
## Different interlace option
## TODO: AND array with 15 instead?
#restoreimage[:,:,::2] = (np.left_shift( halfimage, 4 ) & 15 )
#restoreimage[:,:,1::2] = np.right_shift( halfimage, 4 )
#t_half2 = time.time()
#
#print( "4-byte encoding time (s): %f" % (t_half1 - t_half0) )
#print( "4-byte DEcoding time (s): %f" % (t_half2 - t_half1) )

originalBytes = 50*4096*4096*4 # Guess of size if we saved as float-32


# Something is wrong with lz4hc, it's really not competitive
# All codecs slow down if you use bloscLZ for some reason. blocksize adjustment?

# Because our data is bit-limited by shot noise generally we will always 
# want to use the BITSHUFFLE filter when working with a counting electron detector
# Probably BITSHUFFLE is not so great for floating-point data, however.
SHUFFLE = blosc.BITSHUFFLE
nThreadsList = np.arange( 12, 48+1, 6 )
compressorList = np.array( ['lz4', 'zlib', 'zstd' ] )
clevelList = np.arange(1,7+1)

testRuns = 1
blockSizeList = np.array( [2**15, 2**16, 2**17, 2**18, 2**19, 2**20, 2**21, 2**22], dtype='int64' )

optiShape = [len(doseLevelList), len(compressorList), len(clevelList), 
             len(nThreadsList), len(blockSizeList), testRuns]
cCompress = np.zeros( optiShape )
cBytes = np.zeros( optiShape )
cRatio = np.zeros( optiShape )
cDecompress = np.zeros( optiShape )



for I, doseLevel in enumerate( doseLevelList ):
    for J, compressor in enumerate(compressorList):
        for K, clevel in enumerate( clevelList ):
            for N, N_THREADS in enumerate( nThreadsList ):
                for B, blocksize in enumerate( blockSizeList ):
                    
                    print( "Testing compressor %s level %d with %d threads and blocksize %d at dose level %f counts/pix" % 
                        (compressor, clevel, N_THREADS, blocksize, doseLevel) )
                            
                    for L in np.arange(testRuns):
    
                        t0 = time.time()
                        packedData = doCompression( poissonDataArray[I, ...], 
                                                       compressor=compressor,
                                                       n_threads=N_THREADS, 
                                                       shuffle=SHUFFLE, 
                                                       clevel=clevel,
                                                       blocksize=blocksize )
                        t1 = time.time()
                        cCompress[I,J,K,N,B,L] = t1 - t0
                        for index in np.arange( len(packedData) ):
                            cBytes[I,J,K,N,B,L] += len( packedData[index] )
                        cRatio[I,J,K,N,B,L] = 100.0 * originalBytes / cBytes[I,J,K,N,B,L]
                        
                        t2 = time.time()
                        # Will this work on all frames?  No....
                        result = doDecompression( packedData, len(packedData), N_THREADS )
                        t3 = time.time()
                        cDecompress[I,J,K,N,B,L] = t3 - t2

np.save( "cCompress.npy", cCompress )
np.save( "cBytes.npy", cBytes )
np.save( "cRatio.npy", cRatio )
np.save( "cDecompress.npy", cDecompress )

#print( "4-byte encoding time (s): %f" % (t_half1 - t_half0) )
#print( "4-byte DEncoding time (s): %f" % (t_half2 - t_half1) )
print( "==== Compressor ====" )
print( compressorList )
print( "==== Compression Time (s) ====" )
print( cCompress )
print( "==== DEcompression Time (s) ====" )
print( cDecompress )
print( "==== Compressed Size (MB) ====" )
print( cBytes / 2**20 )
print( "==== Compression Ratio (%) ====" )
print( cRatio )

print( "TODO: test decompression times" )
print( "TODO: test read/write times" )

#print( "Compression time: %f s" %(t1-t0) )
#print( "Original DM4 size: %d MB" % (originalBytes /(2**20)) )
#uncompressedBytes = np.product( dm4image.shape )
#print( "UNcompressed size: %d MB" % (uncompressedBytes/(2**20)) )
#print( "Compressed size: %d MB" % (header['compressedBytes']/(2**20)) )
#print( "Effective compression ratio: %d %% " % (100.0 * originalBytes/header['compressedBytes']) )
