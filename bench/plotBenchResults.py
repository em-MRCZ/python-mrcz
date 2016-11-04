# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:24:09 2016

@author: rmcleod
"""
import numpy as np
import matplotlib.pyplot as plt
import blosc

SHUFFLE = blosc.BITSHUFFLE  # Wow BITSHUFFLE IS FAST
nThreadsList = np.arange( 12, 48+1, 6 )
shuffleList = np.arange(0,2+1)
compressorList = np.array( ['lz4', 'zlib', 'zstd' ] )
clevelList = np.arange(1,6+1)
doseLevelList = [ 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0 ]
testRuns = 1
blockSizeList = np.array( [2**15, 2**16, 2**17, 2**18, 2**19, 2**20, 2**21, 2**22], dtype='int64' )

colorList = ['steelblue', 'forestgreen', 'purple']

optiShape = [len(doseLevelList), len(compressorList), len(clevelList), 
             len(nThreadsList), len(blockSizeList), testRuns]
            
cCompress = np.load( "cCompress.npy"  )
cBytes  = np.load( "cBytes.npy" )
cRatio = np.load( "cRatio.npy" )
cDecompress = np.load( "cDecompress.npy" )

# Average over runs
# (could take std too)
cCompress = np.mean( cCompress, axis=-1 )
cBytes = np.mean( cBytes, axis=-1 )
cRatio = np.mean( cRatio, axis=-1 )
cDecompress = np.mean( cDecompress, axis=-1 )

originalBytes = 50*4096*4096*4 # Guess of size if we saved as float-32

cCompressRate = originalBytes/4 / cCompress / 1E6
cDecompressRate = originalBytes/4 / cDecompress / 1E6

best = np.unravel_index( np.argmax( cCompressRate[3,0,...] ), cCompressRate.shape[2:] )
print( "LZ4 best compression rate: %f at clevel=%d, n_threads=%d, blocksize=%d" % \
      (np.max( cCompressRate[3,0,...] ), clevelList[best[0]], nThreadsList[best[1]], blockSizeList[best[2]]  ) )
best = np.unravel_index( np.argmax( cCompressRate[3,1,...] ), cCompressRate.shape[2:] )
print( "Zlib best compression rate: %f at clevel=%d, n_threads=%d, blocksize=%d" % \
      (np.max( cCompressRate[3,1,...] ), clevelList[best[0]], nThreadsList[best[1]], blockSizeList[best[2]]  ) )
best = np.unravel_index( np.argmax( cCompressRate[3,2,...] ), cCompressRate.shape[2:] )
print( "Zstd best compression rate: %f at clevel=%d, n_threads=%d, blocksize=%d" % \
      (np.max( cCompressRate[3,2,...] ), clevelList[best[0]], nThreadsList[best[1]], blockSizeList[best[2]]  ) )



#for I, doseLevel in enumerate( doseLevelList ):
#    for J, compressor in enumerate(compressorList):
#        for K, clevel in enumerate( clevelList ):
#            for N, N_THREADS in enumerate( nThreadsList ):
#                for B, blocksize in enumerate( blockSizeList ):
    
#### Compression rate ####
maxLz4cRate = np.max( np.max( np.max( cCompressRate[:,0,...], axis=3 ), axis=2 ), axis=1 )
maxZLibcRate = np.max( np.max( np.max( cCompressRate[:,1,...], axis=3 ), axis=2 ), axis=1 )
maxZStdcRate = np.max( np.max( np.max( cCompressRate[:,2,...], axis=3 ), axis=2 ), axis=1 )
maxLz4deRate = np.max( np.max( np.max( cDecompressRate[:,0,...], axis=3 ), axis=2 ), axis=1 )
maxZLibdeRate = np.max( np.max( np.max( cDecompressRate[:,1,...], axis=3 ), axis=2 ), axis=1 )
maxZStddeRate = np.max( np.max( np.max( cDecompressRate[:,2,...], axis=3 ), axis=2 ), axis=1 )
# Something wrong with the decompression benchmark...


uint4CRate = originalBytes/4 / 0.8 / 1E6
plt.figure()
plt.plot( doseLevelList, maxLz4cRate, '.-', 
         linewidth=1.5, markeredgecolor='k', color=colorList[0], label='lz4' )
plt.plot( doseLevelList, maxLz4deRate, '--', color=colorList[0], label='unlz4' )
plt.plot( doseLevelList, maxZLibcRate, '.-',
         linewidth=1.5, markeredgecolor='k', color=colorList[1], label='zlib' )
plt.plot( doseLevelList, maxZLibdeRate, '--', color=colorList[1], label='unzlib' )
plt.plot( doseLevelList, maxZStdcRate, '.-', 
         linewidth=1.5, markeredgecolor='k', color=colorList[2], label='zstd' )
plt.plot( doseLevelList, maxZStddeRate, '--', color=colorList[2], label='unzstd' )
plt.plot( doseLevelList, uint4CRate*np.ones_like(doseLevelList), '-', 
         linewidth=1.5, color='darkorange', label='uint4' )
plt.legend( loc='best' )
plt.xlabel( "Dose ($e^-/pix/frame$)" )
plt.ylabel( "Compression Rate ($MB/s$)" )
plt.savefig( "DoseRate_vs_CompressionRate.png" )

#### Compression ratio ####
maxLz4cRatio = np.max( np.max( np.max( cRatio[:,0,...], axis=3 ), axis=2 ), axis=1 )
maxZLibcRatio = np.max( np.max( np.max( cRatio[:,1,...], axis=3 ), axis=2 ), axis=1 )
maxZStdcRatio = np.max( np.max( np.max( cRatio[:,2,...], axis=3 ), axis=2 ), axis=1 )


plt.figure()
plt.plot( doseLevelList, maxLz4cRatio, '.-', 
         linewidth=1.5, markeredgecolor='k', color=colorList[0], label='lz4' )
plt.plot( doseLevelList, maxZLibcRatio, '.-', 
         linewidth=1.5, markeredgecolor='k', color=colorList[1], label='zlib' )
plt.plot( doseLevelList, maxZStdcRatio, '.-', 
         linewidth=1.5, markeredgecolor='k', color=colorList[2], label='zstd' )
plt.plot( doseLevelList, 800.0 * np.ones_like(doseLevelList), '-', 
         linewidth=1.5, color='darkorange', label='uint4' )
plt.legend( loc='best' )
plt.xlabel( "Dose ($e^-/pix/frame$)" )
plt.ylabel( "Compression Ratio ($\%$)" )
plt.savefig( "DoseRate_vs_CompressionRatio.png" )


# Compression Rate, Ratio, versus clevel
# Just plot for clevel 1?
for I, doseLevel in enumerate( doseLevelList ):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for J, compressor in enumerate( compressorList ):
        
        bestCompressRate = np.zeros_like( clevelList )
        optCRatio = np.zeros_like( clevelList )
        for K in np.arange( len( clevelList ) ):
            bestNB = np.unravel_index( np.argmax( cCompressRate[I,J,K,...] ) , cCompressRate.shape[-2:] )
            bestCompressRate[K] = cCompressRate[I,J,K,bestNB[0],bestNB[1]]
            optCRatio[K] = cRatio[I,J,K,bestNB[0], bestNB[1]]
        plt.semilogx( bestCompressRate, optCRatio, '.-', color=colorList[J], label=compressorList[J],
                 linewidth=1.5, markeredgecolor='k' )
        ax.annotate( '1', xy=(bestCompressRate[0],optCRatio[0]) )
    plt.legend( loc='best' )
    plt.xlabel( "Compression Rate ($MB/s$)" )
    plt.ylabel( "Compression Ratio ($\%$)" )
    plt.title( "Dose level $%.1f e^-/pix/frame$" % doseLevelList[I] )
    plt.savefig( "clevelPlot_dose%f.png" % doseLevelList[I] )
    
    
#### FIND BEST COMPRESSION LEVEL, NTHREADS, BLOCKSIZE FOR COMPRESSIONRATE AND CRATIO             

# I think scaling by the number of threads is a little too confusing?
fig = plt.figure()
I = 3 # 1.0 e-/pix
K = 0
B = 1
for J, compressor in enumerate( compressorList ):
    plt.plot( nThreadsList, cCompressRate[I,J,K,:,B], '.-', color=colorList[J], label=compressorList[J],
             linewidth=1.5, markeredgecolor='k' )
plt.title( "Thread scaling for $1.0 e^-/pix/frame$, clevel $1$, block size $64k$" )
plt.xlabel( "Number of threads" )
plt.ylabel( "Compression Rate ($MB/s$)" )
plt.legend( loc='best' )
plt.savefig( "CompressionRate_threadScaling.png" )


fig = plt.figure()
I = 3 # 1.0 e-/pix
K = 0
B = 1
for J, compressor in enumerate( compressorList ):
    normCompressRate = cCompressRate[I,J,K,:,B] / np.array( nThreadsList )
    plt.plot( nThreadsList, normCompressRate, '.-', color=colorList[J], label=compressorList[J],
             linewidth=1.5, markeredgecolor='k' )
plt.title( "Thread scaling for $1.0 e^-/pix/frame$, clevel $1$, block size $64k$" )
plt.xlabel( "Number of threads" )
plt.ylabel( "Normalized Compression Rate ($MB/s/thread$)" )
plt.legend( loc='best' )
plt.savefig( "NormedCompressionRate_threadScaling.png" )

fig, ax1 = plt.subplots( figsize=(10,8) )
I = 3 # 1.0 e-/pix
K = 0
N = 5
for J, compressor in enumerate( compressorList ):
    ax1.plot( blockSizeList/(2**10), cRatio[I,J,K,N,:], '+-', color=colorList[J], label="Ratio " + compressorList[J],
             linewidth=1.5, markeredgecolor='k' )
ax1.set_xlabel( "Block size ($kB$)" )
ax1.set_ylabel( "Compression Ratio ($\%$)" )
ax1.legend( loc='center' )
ax2 = ax1.twinx()
for J, compressor in enumerate( compressorList ):
    ax2.plot( blockSizeList/(2**10), cCompressRate[I,J,K,N,:], 'o--', color=colorList[J], label="Rate " + compressorList[J],
             linewidth=1.5, markeredgecolor='k' )
ax2.set_ylabel( "Compression Rate ($MB/s$)" )
plt.title( "Scaling with block size" )
ax2.legend( loc='right' )
plt.savefig( "blockScaling.png" )



#plt.figure()
#plt.plot( doseLevelList, compressStackSize, 'k.-'  )
#plt.xlabel( r"Dose Rate $(e^-/\AA^2)$" )
#plt.ylabel( r"Compressed file size (GB)" )
#plt.title( r"Expected stack size for a total dose of $80 e^-/\AA^2 (zstd5)$" )
#plt.savefig( "ExpectedMRCZSizeFor80e.png" )