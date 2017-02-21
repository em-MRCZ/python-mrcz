# -*- coding: utf-8 -*-
"""
Python Utilities for Relion

Created on Tue Dec  1 14:26:13 2015
@author: Robert A. McLeod
@email: robbmcleod@gmail.com OR robert.mcleod@unibas.ch

This is a primarily a general parser for Relion star files.  It creates a two-level dictionary, with the 
"data_*" level at the top and the "_rln*" level at the second. Use the star.keys() function to see what values 
the dictionary has.  I.e.
    
    rln.star.keys()

and then

    rln.star['data_whatever'].keys()

Example usage:

    rln = ReliablePy()
    # Wildcards can be loaded
    rln.load( 'PostProcess*.star' )
    
    # Plot the Fourier Shell Correlation
    plt.figure()
    plt.plot( rln.star['data_fsc']['Resolution'], rln.star['data_fsc']['FourierShellCorrelationUnmaskedMaps'], '.-' )
    plt.xlabel( "Resolution" )
    plt.ylabel( "FSC" )
    
Note that all Relion strings are byte-strings (char1) rather than UTF encoded.

"""
from __future__ import division, print_function, absolute_import


from . import ioDM, ioMRC

import numpy as np
import os, os.path
import glob
import time

from collections import OrderedDict

# The following are not requirements of python-mrcz, only ReliablePy:
import matplotlib.pyplot as plt
import scipy
import pandas

# Static variable decorator
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate
    
    
def apodization( name = 'butter.32', shape= [2048,2048], radius=None ):
    """ apodization( name = 'butter.32', size = [2048,2048], radius=None )
    Provides a 2-D filter or apodization window for Fourier filtering or image clamping.
        Radius = None defaults to shape/2
    
    Valid names are: 
        'hann' - von Hann cosine window on radius
        'hann_square' as above but on X-Y
        'hamming' - good for apodization, nonsense as a filter
        'butter.X' Butterworth multi-order filter where X is the order of the Lorentzian
        'butter_square.X' Butterworth in X-Y
        'gauss_trunc' - truncated gaussian, higher performance (smaller PSF) than hann filter
        'gauss' - regular gaussian
    NOTE: There are windows in scipy.signal for 1D-filtering...
    WARNING: doesn't work properly for odd image dimensions
    """
    # Make meshes
    shape = np.asarray( shape )
    if radius is None:
        radius = shape/2.0
    else:
        radius = np.asarray( radius, dtype='float' )
    # DEBUG: Doesn't work right for odd numbers
    [xmesh,ymesh] = np.meshgrid( np.arange(-shape[1]/2,shape[1]/2), np.arange(-shape[0]/2,shape[0]/2) )
    r2mesh = xmesh*xmesh/( np.double(radius[0])**2 ) + ymesh*ymesh/( np.double(radius[1])**2 )
    
    try:
        [name, order] = name.lower().split('.')
        order = np.double(order)
    except ValueError:
        order = 1
        
    if name == 'butter':
        window =  np.sqrt( 1.0 / (1.0 + r2mesh**order ) )
    elif name == 'butter_square':
        window = np.sqrt( 1.0 / (1.0 + (xmesh/radius[1])**order))*np.sqrt(1.0 / (1.0 + (ymesh/radius[0])**order) )
    elif name == 'hann':
        cropwin = ((xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') * 0.5 * ( 1.0 + np.cos( 1.0*np.pi*np.sqrt( (xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0  )  ) )
    elif name == 'hann_square':
        window = ( (0.5 + 0.5*np.cos( np.pi*( xmesh/radius[1]) ) ) *
            (0.5 + 0.5*np.cos( np.pi*( ymesh/radius[0] )  ) ) )
    elif name == 'hamming':
        cropwin = ((xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') *  ( 0.54 + 0.46*np.cos( 1.0*np.pi*np.sqrt( (xmesh/radius[1])**2.0 + (ymesh/radius[0])**2.0  )  ) )
    elif name == 'hamming_square':
        window = ( (0.54 + 0.46*np.cos( np.pi*( xmesh/radius[1]) ) ) *
            (0.54 + 0.46*np.cos( np.pi*( ymesh/radius[0] )  ) ) )
    elif name == 'gauss' or name == 'gaussian':
        window = np.exp( -(xmesh/radius[1])**2.0 - (ymesh/radius[0])**2.0 )
    elif name == 'gauss_trunc':
        cropwin = ((0.5*xmesh/radius[1])**2.0 + (0.5*ymesh/radius[0])**2.0) <= 1.0
        window = cropwin.astype('float') * np.exp( -(xmesh/radius[1])**2.0 - (ymesh/radius[0])**2.0 )
    elif name == 'lanczos':
        print( "TODO: Implement Lanczos window" )
        return
    else:
        print( "Error: unknown filter name passed into apodization" )
        return
    return window
 
def pyFFTWPlanner( realMage, fouMage=None, wisdomFile = None, effort = 'FFTW_MEASURE', n_threads = None, doForward = True, doReverse = True ):
    """
    Appends an FFTW plan for the given realMage to a text file stored in the same
    directory as RAMutil, which can then be loaded in the future with pyFFTWLoadWisdom.
    
    NOTE: realMage should be typecast to 'complex64' normally.
    
    NOTE: planning pickle files are hardware dependant, so don't copy them from one 
    machine to another. wisdomFile allows you to specify a .pkl file with the wisdom
    tuple written to it.  The wisdomFile is never updated, whereas the default 
    wisdom _is_ updated with each call. For multiprocessing, it's important to 
    let FFTW generate its plan from an ideal processor state.
    
    TODO: implement real, half-space fourier transforms rfft2 and irfft2 as built
    """
    
    import pyfftw
    import pickle
    import os.path
    from multiprocessing import cpu_count
    
    utilpath = os.path.dirname(os.path.realpath(__file__))
    
    # First import whatever we already have
    if wisdomFile is None:
        wisdomFile = os.path.join( utilpath, "pyFFTW_wisdom.pkl" )


    if os.path.isfile(wisdomFile):
        try:
            fh = open( wisdomFile, 'rb')
        except:
            print( "Util: pyFFTW wisdom plan file: " + str(wisdomFile) + " invalid/unreadable" )
            
        try:
            pyfftw.import_wisdom( pickle.load( fh ) )
        except: 
            # THis is not normally a problem, it might be empty?
            print( "Util: pickle failed to import FFTW wisdom" )
            pass
        try:
            fh.close()
        except: 
            pass

    else:
        # Touch the file
        os.umask(0000) # Everyone should be able to delete scratch files
        with open( wisdomFile, 'wb') as fh:
            pass
    
        # I think the fouMage array has to be smaller to do the real -> complex FFT?
    if fouMage is None:
        if realMage.dtype.name == 'float32':
            print( "pyFFTW is recommended to work on purely complex data" )
            fouShape = realMage.shape
            fouShape.shape[-1] = realMage.shape[-1]//2 + 1
            fouDtype =  'complex64'
            fouMage = np.empty( fouShape, dtype=fouDtype )
        elif realMage.dtype.name == 'float64': 
            print( "pyFFTW is recommended to work on purely complex data" )
            fouShape = realMage.shape
            fouShape.shape[-1] = realMage.shape[-1]//2 + 1
            fouDtype = 'complex128'
            fouMage = np.empty( fouShape, dtype=fouDtype )
        else: # Assume dtype is complexXX
            fouDtype = realMage.dtype.name
            fouMage = np.zeros( realMage.shape, dtype=fouDtype )
            
    if n_threads is None:
        n_threads = cpu_count()
    print( "FFTW using " + str(n_threads) + " threads" )
    
    if bool(doForward):
        #print( "Planning forward pyFFTW for shape: " + str( realMage.shape ) )
        FFT2 = pyfftw.builders.fft2( realMage, planner_effort=effort, 
                                    threads=n_threads, auto_align_input=True )
    else:
        FFT2 = None
    if bool(doReverse):
        #print( "Planning reverse pyFFTW for shape: " + str( realMage.shape ) )
        IFFT2 = pyfftw.builders.ifft2( fouMage, planner_effort=effort, 
                                      threads=n_threads, auto_align_input=True )
    else: 
        IFFT2 = None

    # Setup so that we can call .execute on each one without re-copying arrays
    # if FFT2 is not None and IFFT2 is not None:
    #    FFT2.update_arrays( FFT2.get_input_array(), IFFT2.get_input_array() )
    #    IFFT2.update_arrays( IFFT2.get_input_array(), FFT2.get_input_array() )
    # Something is different in the builders compared to FFTW directly. 
    # Can also repeat this for pyfftw.builders.rfft2 and .irfft2 if desired, but 
    # generally it seems slower.
    # Opening a file for writing is supposed to truncate it
    # if bool(savePlan):
    #if wisdomFile is None:
    # with open( utilpath + "/pyFFTW_wisdom.pkl", 'wb') as fh:
    with open( wisdomFile, 'wb' ) as fh:
        pickle.dump( pyfftw.export_wisdom(), fh )
            
    return FFT2, IFFT2
    
# TODO: put IceFilter in a ReliablePy utility function file
@static_var( "bpFilter", -1 )
@static_var( "mageShape", np.array([0,0]) )
@static_var( "ps", -42 )
@static_var( "FFT2", -42 )
@static_var( "IFFT2", -42 )
def IceFilter( mage, pixelSize=1.0, filtRad = 8.0 ):
    """
    IceFilter applies a band-pass filter to mage that passes the first 3 
    water ice rings, and then returns the result.
        pixelSize is in ANGSTROMS because this is bio.  Program uses this to 
        calculate the width of the band-pass filter.
        filtRad is radius of the Gaussian filter (pixels) to apply after Fourier filtration 
        that are periodic artifacts due to multiple defocus zeros being in the band 
    """
    
    # First water ring is at 3.897 Angstroms
    # Second is ater 3.669 Angstroms
    # Third is at 3.441 Angstroms
    # And of course there is strain, so go from about 4 to 3.3 Angstroms in the mesh
    # Test for existance of pyfftw
    try:
        import pyfftw
        pyfftwFound = True
    except:
        pyfftwFound = False

    # Check to see if we have to update our static variables
    if ( (IceFilter.mageShape != mage.shape).any() ) or (IceFilter.bpFilter.size == 1) or (IceFilter.ps != pixelSize):
        # Make a new IceFilter.bpFilter
        IceFilter.mageShape = np.array( mage.shape )
        IceFilter.ps = pixelSize
        
        bpMin = pixelSize / 4.0  # pixels tp the 4.0 Angstrom spacing
        bpMax = pixelSize / 3.3  # pixels to the 3.3 Angstrom spacing
        
        # So pixel frequency is -0.5 to +0.5 with shape steps
        # And we want a bandpass from 1.0/bpMin to 1.0/bpMax, which is different on each axis for rectangular images
        pixFreqX = 1.0 / mage.shape[1]
        pixFreqY = 1.0 / mage.shape[0]
        bpRangeX = np.round( np.array( [ bpMin/pixFreqX, bpMax/pixFreqX ] ) )
        bpRangeY = np.round( np.array( [ bpMin/pixFreqY, bpMax/pixFreqY ] ) )
        IceFilter.bpFilter = np.fft.fftshift( 
            (1.0 - apodization( name='butter.64', size=mage.shape, radius=[ bpRangeY[0],bpRangeX[0] ] )) 
            * apodization( name='butter.64', size=mage.shape, radius=[ bpRangeY[1],bpRangeX[1] ] ) )
        IceFilter.bpFilter = IceFilter.bpFilter.astype( 'float32' ) 
        
        if pyfftwFound: [IceFilter.FFT2, IceFilter.IFFT2] = pyFFTWPlanner( mage.astype('complex64') )
        pass
    
    # Apply band-pass filter
    if pyfftwFound:
        IceFilter.FFT2.update_arrays( mage.astype('complex64'), IceFilter.FFT2.get_output_array() )
        IceFilter.FFT2.execute()
        IceFilter.IFFT2.update_arrays( IceFilter.FFT2.get_output_array() * IceFilter.bpFilter, IceFilter.IFFT2.get_output_array() )
        IceFilter.IFFT2.execute()
        bpMage = IceFilter.IFFT2.get_output_array() / mage.size
    else:
        FFTmage = np.fft.fft2( mage )
        bpMage = np.fft.ifft2( FFTmage * IceFilter.bpFilter )

    from scipy.ndimage import gaussian_filter
    bpGaussMage = gaussian_filter( np.abs(bpMage), filtRad )
    # So if I don't want to build a mask here, and if I'm just doing band-pass
    # intensity scoring I don't need it, I don't need to make a thresholded mask
    
    # Should we normalize the bpGaussMage by the mean and std of the mage?
    return bpGaussMage

class ReliablePy(object):
    
    def __init__( self, *inputs ) :
        self.verbose = 1
        self.inputs = list( inputs )
        
        # _data.star file dicts
        self.star = OrderedDict()
        
        self.par = []
        self.pcol = OrderedDict()

        self.box = [] # Each box file loaded is indexed by its load order / dict could also be done if it's more convienent.
        
        # Particle/class data
        self.mrc = []
        self.mrc_header = []
        if inputs:
            self.load( *inputs )
        pass
    
    def load( self, *input_names ):
        # See if it's a single-string or list/tuple

        if not isinstance( input_names, str ):
            new_files = []
            for item in input_names:
                new_files.extend( glob.glob( item ) )
        else:
            new_files = list( input_names )
        
        for filename in new_files:
            [fileFront, fileExt] = os.path.splitext( filename )
            
            if fileExt == '.mrc' or fileExt == '.mrcs':
                self.inputs.append(filename)
                self.__loadMRC( filename )
            elif fileExt == '.star':
                self.inputs.append(filename)
                self.__loadStar( filename )
            elif fileExt == '.par':
                self.inputs.append(filename)
                self.__loadPar( filename )
            elif fileExt == '.box':
                self.inputs.append(filename)
                self.__loadBox( filename )
            else:
                print( "Unknown file extension passed in: " + filename )
                
    def plotFSC( self ):
        # Do error checking?  Or no?
        plt.rc('lines', linewidth=2.0, markersize=12.0 )
        plt.figure()
        plt.plot( self.star['data_fsc']['Resolution'], 0.143*np.ones_like(self.star['data_fsc']['Resolution']), 
                 '-', color='firebrick', label="Resolution criteria" )
        try:
            plt.plot( self.star['data_fsc']['Resolution'], self.star['data_fsc']['FourierShellCorrelationUnmaskedMaps'], 
                 'k.-', label="Unmasked FSC" )
        except: pass
        try:
            plt.plot( self.star['data_fsc']['Resolution'], self.star['data_fsc']['FourierShellCorrelationMaskedMaps'], 
                 '.-', color='royalblue', label="Masked FSC" )   
        except: pass
        try:         
            plt.plot( self.star['data_fsc']['Resolution'], self.star['data_fsc']['FourierShellCorrelationCorrected'], 
                 '.-', color='forestgreen', label="Corrected FSC" )          
        except: pass
        try:
            plt.plot( self.star['data_fsc']['Resolution'], self.star['data_fsc']['CorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps'], 
                 '.-', color='goldenrod', label="Random-phase corrected FSC" )
        except: pass
        plt.xlabel( "Resolution ($\AA^{-1}$)" )
        plt.ylabel( "Fourier Shell Correlation" )
        plt.legend( loc='upper right', fontsize=16 )
        plt.xlim( np.min(self.star['data_fsc']['Resolution']), np.max(self.star['data_fsc']['Resolution']) )
        print( "Final resolution (unmasked): %.2f A"%self.star['data_general']['FinalResolution']  )
        print( "B-factor applied: %.1f"%self.star['data_general']['BfactorUsedForSharpening'] )
        
    def plotSSNR( self ):
        """
        Pulls the SSNR from each class in a _model.star file and plots them, for assessing which class is the 
        'best' class 
        """
        
        N_particles = np.sum( self.star[b'data_model_groups'][b'GroupNrParticles'] )
        N_classes = self.star[b'data_model_general'][b'NrClasses']
            
        plt.figure()
        for K in np.arange( N_classes ):
            Resolution = self.star[b'data_model_class_%d'%(K+1)][b'Resolution']
            SSNR = self.star[b'data_model_class_%d'%(K+1)][b'SsnrMap']
            plt.semilogy( Resolution, SSNR+1.0, 
                     label="Class %d: %d" %(K+1,N_particles*self.star[b'data_model_classes'][b'ClassDistribution'][K]) )
        plt.legend( loc = 'best' )
        plt.xlabel( "Resolution ($\AA^{-1}$)" )
        plt.ylabel( "Spectral Signal-to-Noise Ratio" )
        # Let's also display the class distributions in the legend
        
    
    def pruneParticlesNearImageEdge( self, box = None, shapeImage = [3838,3710] ):
        """
        Removes any particles near image edge. Relion's default behavoir is to replicate pad these, 
        which often leads to it crashing.
        
        box is the bounding box size for the particle, in pixels.  If a _model.star file is loaded 
        it is automatically detected.  Otherwise it must be provided.
        
        Image size is not stored anywhere obvious in Relion, so it must be passed in in terms of 
        it's shape in [y,x]
        """
        if box == None:
            try: 
                box = self.star[b'data_model_general'][b'OriginalImageSize']
            except:
                print( "No box shape found in metadata, load a *_model.star file or provide box dimension" )
                return
            
        partCount = len( self.star[b'data_'][b'CoordinateX'] )
        
        # Hmm... removing a row is a little painful because I index by keys in columnar format.
        box2 = box/2
        CoordX = self.star[b'data_'][b'CoordinateX']
        CoordY = self.star[b'data_'][b'CoordinateY']
        keepElements = ~((CoordX < box2)|(CoordY < box2)|(CoordX > shapeImage[1]-box2)|(CoordY > shapeImage[0]-box2))
        for key, store in self.star[b'data_'].items():
            self.star[b'data_'][key] = store[keepElements]
        print( "Deleted %d"%(partCount-len(self.star[b'data_'][b'CoordinateX']) ) + 
                " particles too close to image edge" )
        pass
    
    def permissiveMask( self, volumeThres, gaussSigma = 5.0, gaussRethres = 0.07, smoothSigma=1.5 ):
        """
        Given a (tight) volumeThres(hold) measured in Chimera or IMS, this function generates a 
        Gaussian dilated mask that is then smoothed.  Everything is done with Gaussian operations 
        so the Fourier space representation of the mask should be relatively smooth as well, 
        and hence ring less.
        
        Excepts self.mrc to be loaded.  Populates self.mask.  
        
        """
        thres = self.mrc > volumeThres; thres = thres.astype('float32')
        
        gaussThres = scipy.ndimage.gaussian_filter( thres, gaussSigma )
        rethres = gaussThres > gaussRethres; rethres = rethres.astype('float32')
        
        self.mask = scipy.ndimage.gaussian_filter( rethres, smoothSigma )
        print( "permissive mask complete, use ioMRC.writeMRC(self.mrc, 'maskname.mrc') to save" )
        pass
    
    def box2star( self, directory = "." ):
        """
        Converts all EMAN .box files in a directory to the associated .star files. Relion cannot successfully 
        rescale particles if they come in .box format.  Also does box pruning if they are too close to an edge.
        """

        boxList = glob.glob( os.path.join( directory, "*.box") )

        starHeader = """
        data_
        
        loop_
        _rlnCoordinateX #1
        _rlnCoordinateY #2
        """
        
        shapeImage = [3838,3710]
        for boxFile in boxList:
            print( "Loading %s" % boxFile )
            boxData = np.loadtxt(boxFile)
            
            xCoord = boxData[:,0]
            yCoord = boxData[:,1]
            boxX = boxData[:,2]/2
            boxY = boxData[:,3]/2
            
            keepElements = ~((xCoord < boxX)|(yCoord < boxY)|(xCoord > shapeImage[1]-boxX)|(yCoord> shapeImage[0]-boxY))
            xCoord = xCoord[keepElements]
            yCoord = yCoord[keepElements]
            boxX = boxX[keepElements]
            boxY = boxY[keepElements]
            
            starFilename = os.path.splitext( boxFile )[0] + ".star"
            with open( starFilename, 'wb' ) as sh:
                sh.writelines( starHeader )
                for J in np.arange(0,len(xCoord)):
                    sh.write( "%.1f %.1f\n" % (xCoord[J]+boxX[J], yCoord[J]+boxY[J] ) )
                   
                sh.write( "\n" )
            sh.close()
            
    def regroupKmeans( self, partPerGroup = 100, miniBatch=True ):
        """
        Does a 3-D k-means clustering on DefocusU, DefocusV, and GroupScaleCorrection
        
        partPerGroup is a suggestion, that is the number of groups is the # of particles / partPerGroup, 
        so outlier groups will tend to have far fewer particle counts that those in the bulk of the data.
        
        miniBatch=True is faster for very large sets (>100,000 particles), but somewhat less accurate
        miniBatch=False is faster for smaller sets, and better overall
        """
        # K-means clustering
        import sklearn
        import sklearn.cluster
        
        # We need to make an array for all particles that has the GroupScaleCorrection
        P = len( self.star[b'data_'][b'DefocusU'] )
        n_clusters = np.int( P / partPerGroup )
        
        DefocusU = self.star[b'data_'][b'DefocusU']
        DefocusV = self.star[b'data_'][b'DefocusV']
        DefocusMean = 0.5* (DefocusU + DefocusV)
        
        if b'data_model_groups' in self.star:
            SCALE_CORR_PRESENT = True
            part_GroupScaleCorrection = np.zeros_like( self.star[b'data_'][b'DefocusU'] )
        
            # Build a GroupScaleCorrection vector
            for J, groupNr in enumerate( self.star[b'data_'][b'GroupNumber'] ):
                part_GroupScaleCorrection[J] = self.star[b'data_model_groups'][b'GroupScaleCorrection'][  np.argwhere(self.star[b'data_model_groups'][b'GroupNumber'] == groupNr)[0] ]
        else:
            print( "No _model.star loaded, not using scale correction" )
            SCALE_CORR_PRESENT = False
            
        ##################
        # K-means clustering:
        ##################
        print( "Running K-means clustering analysis for " + str(P) + " particles into " + str(n_clusters) + " clusters" )
        t0 = time.time()
        if bool(miniBatch):
            print( "TODO: determine number of jobs for K-means" )
            k_means = sklearn.cluster.MiniBatchKMeans( n_clusters=n_clusters, init_size=3*n_clusters+1 )
        else: 
            k_means = sklearn.cluster.KMeans( n_clusters=n_clusters, n_jobs=12 )
        #Kmeans_in = np.vstack( [DefocusMean, part_GroupScaleCorrection]).transpose()
        if SCALE_CORR_PRESENT:
            Kmeans_in = np.vstack( [DefocusU,DefocusV, part_GroupScaleCorrection]).transpose()
        else:
            Kmeans_in = np.vstack( [DefocusU,DefocusV]).transpose()
        Kmeans_in = sklearn.preprocessing.robust_scale( Kmeans_in )
        k_predict = k_means.fit_predict( Kmeans_in  )
        t1 = time.time()
        print( "Cluster analysis finished in (s): " + str(t1-t0) )
        
        if self.verbose >= 2:
            plt.figure()
            plt.scatter( DefocusMean, part_GroupScaleCorrection, c=k_predict)
            plt.xlabel( "Defocus ($\AA$)" )
            plt.ylabel( "Group scale correction (a.u.)" )
            plt.title("K-means on Defocus")
        
        ##################
        # Save the results in a new particles .star file:
        ##################
        # Replace, add one to group number because Relion starts counting from 1

        particleKey = b"data_"
        
        # Add the GroupName field to the star file
        self.star[particleKey][b'GroupName'] = [""] * len( self.star[particleKey][b'GroupNumber'] )
        for J, groupName in enumerate( k_predict ):
            self.star[particleKey][b'GroupName'][J] = b'G' + str(groupName + 1)
            
        # Build a new group number count
        groupCount = np.zeros_like( self.star[particleKey][b'GroupNumber'] )
        for J in np.arange(0,len(groupCount)):
            groupCount[J] = np.sum( self.star[particleKey][b'GroupNumber'] == J )
        self.star[particleKey][b'GroupNumber'] = groupCount
            
        # Recalculate number of particles in each group (ACTUALLY THIS SEEMS NOT NECESSARY)
        #GroupNr = np.zeros( np.max( k_predict )+1 )
        #for J in xrange( np.min( k_predict), np.max( k_predict ) ):
        #    GroupNr[J] = np.sum( k_predict == J )
        #    pass
        #
        #for J in xrange(0, len(rln.star[particleKey]['GroupNumber']) ):
        #    rln.star[particleKey]['GroupNumber'][J] = GroupNr[ k_predict[J] ]
            
    def saveDataStar( self, outputName, particleKey = b"data_" ):
        """
        Outputs a relion ..._data.star file that has been pruned, regrouped, etc. to outputName
        """
        
        if outputName == None:
            # Need to store input star names, and figure out which was the last loaded particles.star file.
            # [outFront, outExt] = os.path.splitext()
            raise IOError( "Default filenames for saveDataStar not implemented yet" )
            
        # TODO: more general star file output    
        # Let's just hack this
        fh = open( outputName, 'wb' )
        fh.write( b"\ndata_\n\nloop_\n")
        
        # Since we made self.star an OrderedDict we don't need to keep track of index ordering
        
        headerKeys = self.star[particleKey].keys()
        

        for J, key in enumerate(headerKeys):
            # print( "Column: " + "_rln" + lookupDict[J+1] + " #" + str(J+1) )
            fh.write( b"_rln" + key + " #" + str(J) + "\n")

        
        # lCnt = len( headerKeys ) 
        P = len( self.star[particleKey][ self.star[particleKey].keys()[0] ] )
        for I in np.arange(0,P):
            fh.write( b"    ")
            for J, key in enumerate(headerKeys):
                fh.write( str( self.star[particleKey][key][I] ) )
                fh.write( b"   " )
            fh.write( b"\n" )
        fh.close()
        
    def saveDataAsPar( self, outputPrefix, N_classes = 1, mag = None, pixelsize=None, particleKey = "data_" ):
        """
        Saves a Relion .star file as a Frealign .par meta-data file.  Also goes through all the particles in the 
        Relion .star and generates an appropriate meta-MRC particle file for Frealign.
        
        Usage:
            
            saveDataAsPar( self, outputPrefix, N_classes = 1, mag = None, pixelsize=None, particleKey = "data_" )
        
            outputPrefix will be appended with "_1_rX.par", where X is the class number.
            
            N_classes will generate N classes with random occupancy, or 100.0 % occupancy for one class.
            
            mag wil change the Relion magnification to the given integer.
            
            pixelsize is also optional.  Relion tends to have round-off error in the pixelsize.
        
        Use 'relion_stack_create --i particles.star --o forFrealign' to generate the associated mrc file.
        
        Also no comment lines are written to the .par file.
        """
        

        
        partCount = len( self.star[b'data_'][b'MicrographName'] )
        # Need AnglePsi, AngleTilt, and AngleRot
        if not b'AnglePsi' in self.star[b'data_']:
            self.star[b'data_'][b'AnglePsi'] = np.zeros( partCount, dtype='float32' )
        if not b'AngleTilt' in self.star[b'data_']:
            self.star['data_'][b'AngleTilt'] = np.zeros( partCount, dtype='float32' )
        if not b'AngleRot' in self.star[b'data_']:
            self.star[b'data_'][b'AngleRot'] = np.zeros( partCount, dtype='float32' )
        if not b'OriginY' in self.star[b'data_']:
            self.star[b'data_'][b'OriginY'] = np.zeros( partCount, dtype='float32' )
        if not b'OriginX' in self.star[b'data_']:
            self.star[b'data_'][b'OriginX'] = np.zeros( partCount, dtype='float32' )
        if not b'Magnification' in self.star[b'data_']:
            self.star[b'data_'][b'Magnification'] = np.zeros( partCount, dtype='float32' )     
        if not b'GroupNumber' in self.star[b'data_']:
            self.star[b'data_'][b'GroupNumber'] = np.zeros( partCount, dtype='uint16' )    
        if not b'DefocusU' in self.star[b'data_']:
            self.star[b'data_'][b'DefocusU'] = np.zeros( partCount, dtype='float32' )
        if not b'DefocusV' in self.star[b'data_']:
            self.star[b'data_'][b'DefocusV'] = np.zeros( partCount, dtype='float32' )
        if not b'DefocusAngle' in self.star[b'data_']:
            self.star[b'data_'][b'DefocusAngle'] = np.zeros( partCount, dtype='float32' )
            

        # Frealign expects shifts in Angstroms.  Pixelsize is sort of sloppily 
        # kept track of in Relion with Magnification and DetectorPixelSize (which
        # defaults to 14.0)
        
        if pixelsize == None:
            # Detector pixel size in um, we need pixelsize in Angstrom
            pixelsize = self.star[b'data_'][b'DetectorPixelSize'][0]*1E4 / self.star[b'data_'][b'Magnification'][0]
            print( "Found pixelsize of %0.f" % pixelsize )
        if mag == None:
            print( "Using Relion magnification of %.f and DSTEP=%.1f" % ( self.star[b'data_'][b'Magnification'][0], self.star[b'data_'][b'DetectorPixelSize'][0]) )
            print( "For a K2 (DSTEP=5.0) the appropriate magnification would be %0.f" % 50000/pixelsize )
        else:
            self.star[b'data_'][b'Magnification'] = mag * np.ones_like( self.star[b'data_'][b'Magnification'] )
        
        logP = int(-500)
        sigma = 1.0
        score = 20.0
        change = 0.0 
        for K in np.arange( 1, N_classes+1 ):
            outputName = outputPrefix + "_1_r%d.par" % K
            if N_classes > 1:
                # Add random occupancy
                occupancy = np.random.uniform( low=0.0, high=100.0, size=len(self.star[b'data_'][b'DefocusU']) )
            else:
                occupancy = 100.0* np.ones_like( self.star[b'data_'][b'DefocusU'] )
            
            with open( outputName, 'w' ) as fh:
                # Frealign is very picky about the number of digits, see card10.f, line 163
                #READ(LINE,*,ERR=99,IOSTAT=CNT)ILIST(NANG),
                #     +                  PSI(NANG),THETA(NANG),PHI(NANG),SHX(NANG),
                #     +                  SHY(NANG),ABSMAGP,FILM(NANG),DFMID1(NANG),
                #     +                  DFMID2(NANG),ANGAST(NANG),OCC(NANG),
                #     +                  LGP,SIG(NANG),PRESA(NANG)
                #7011                FORMAT(I7,3F8.2,2F10.2,I8,I6,2F9.1,2F8.2,I10,F11.4,
                #     +                     F8.2)
                for J in np.arange(partCount):
                    fh.write( "%7d"%(J+1) 
                                + " %8.2f"%self.star[b'data_'][b'AnglePsi'][J] 
                                + " %8.2f"%self.star[b'data_'][b'AngleTilt'][J]  
                                + " %8.2f"%self.star[b'data_'][b'AngleRot'][J] 
                                + " %8.2f"%(self.star[b'data_'][b'OriginX'][J] * pixelsize)
                                + " %8.2f"%(self.star[b'data_'][b'OriginY'][J] * pixelsize)
                                + " %8.0f"%self.star[b'data_'][b'Magnification'][J] 
                                + " %6d"%self.star[b'data_'][b'GroupNumber'][J] 
                                + " %9.1f"%self.star[b'data_'][b'DefocusU'][J]
                                + " %9.1f"%self.star[b'data_'][b'DefocusV'][J] 
                                + " %8.2f"%self.star[b'data_'][b'DefocusAngle'][J] 
                                + " %8.2f"%occupancy[J] 
                                + " %10d"%logP
                                + " %11.4f"%sigma
                                + " %8.2f"%score
                                + " %8.2f"%change + "\n")
                pass
        
        # Ok and now we need to make a giant particles file?
        #mrcName, _= os.path.splitext( outputName ) 
        #mrcName = mrcName + ".mrc"
        
        #imageNames = np.zeros_like( self.star[b'data_'][b'ImageName'] )
        #for J, name in enumerate( self.star[b'data_'][b'ImageName'] ):
        #    imageNames[J] = name.split('@')[1]
        #uniqueNames = np.unique( imageNames ) # Ordering is preserved, thankfully!
        

        # It would be much better if we could write to a memory-mapped file rather than building the entire array in memory
        # However this is a little buggy in numpy.
        # https://docs.python.org/2/library/mmap.html instead?
        
        #particleList = []
        #for uniqueName in uniqueNames:
        #   particleList.extend( ioMRC.readMRC(uniqueName)[0] )
            
        #print( "DONE building particle list!" )
        #print( len(particleList) )
        
        #particleArray = np.array( particleList )
        # del particleList
        
        # We do have the shape parameter that we can pass in to pre-pad the array with all zeros.
        #ioMRC.writeMRC( particleArray, mrcName, shape=None ) # TODO: no pixelsize
        pass
        
    def saveCtfImagesStar( self, outputName, zorroList = "*.dm4.log", physicalPixelSize=5.0, amplitudeContrast=0.08 ):
        """
        Given a glob pattern, generate a list of zorro logs, or alternatively one can pass in a list.  For each
        zorro log, load it, extract the pertinant info (defocus, etc.).  This is a file ready for particle 
        extraction, with imbedded Ctf information.
        """
        import zorro
        
        zorroList = glob.glob( zorroList )

        headerDict = { b'MicrographName':1, b'CtfImage':2, b'DefocusU':3, b'DefocusV':4, b'DefocusAngle':5, 
                      b'Voltage':6, b'SphericalAberration':7, b'AmplitudeContrast':8, b'Magnification':9, 
                      b'DetectorPixelSize':10, b'CtfFigureOfMerit': 11 }
        lookupDict = dict( zip( headerDict.values(), headerDict.keys() ) )   
        data = OrderedDict()
        for header in headerDict:      
            data[header] = [None]*len(zorroList)
            
        zorroReg = zorro.ImageRegistrator()    
        for J, zorroLog in enumerate(zorroList):
            zorroReg.loadConfig( zorroLog, loadData=False )
            
            data[b'MicrographName'][J] = zorroReg.files['sum']
            data[b'CtfImage'][J] = os.path.splitext( zorroReg.files['sum'] )[0] + ".ctf:mrc"
            # CTF4Results = [Micrograph number, DF1, DF2, Azimuth, Additional Phase shift, CC, max spacing fit-to]
            data[b'DefocusU'][J] = zorroReg.CTF4Results[1]
            data[b'DefocusV'][J] = zorroReg.CTF4Results[2]
            data[b'DefocusAngle'][J] = zorroReg.CTF4Results[3]
            data[b'CtfFigureOfMerit'][J] = zorroReg.CTF4Results[5]
            data[b'Voltage'][J] = zorroReg.voltage
            data[b'SphericalAberration'][J] = zorroReg.C3
            data[b'AmplitudeContrast'][J] = amplitudeContrast
            data[b'DetectorPixelSize'][J] = physicalPixelSize
            data[b'Magnification'][J] = physicalPixelSize / (zorroReg.pixelsize * 1E-3)
            
        with open( outputName, 'wb' ) as fh:

            fh.write( b"\ndata_\n\nloop_\n")
            for J in np.sort(lookupDict.keys()):
                # print( "Column: " + "_rln" + lookupDict[J+1] + " #" + str(J+1) )
                fh.write( b"_rln" + lookupDict[J] + b" #" + str(J) + b"\n")
            
            lCnt = len( lookupDict ) 
            for I in np.arange(0,len(zorroList)):
                fh.write( b"    ")
                for J in np.arange(0,lCnt):
                    fh.write( str( data[lookupDict[J+1]][I] )  )
                    fh.write( b"   " )
                fh.write( b"\n" )
            
    def gctfHistogramFilter( self, defocusThreshold = 40000, astigThreshold = 800, 
                            fomThreshold = 0.0, resThreshold = 6.0, 
                            starName = "micrographs_all_gctf.star", outName = "micrographs_pruned_gctf.star" ):
        """
        gctfHistogramFilter( self, defocusThreshold = 40000, astigThreshold = 800, 
                            fomThreshold = 0.0, resThreshold = 6.0, 
                            starName = "micrographs_all_gctf.star", outName = "micrographs_pruned_gctf.star" )
                            
        Calculates histograms of defocus, astigmatism, figure-of-merit (Pearson correlation coefficient),
        and resolution limit, and applies the thresholds as specified in the keyword arguments.  
        Plots are generated showing the threshold level.  
        
        The output star file `outName` rejects all micrographs that fail any of the thresholds.
        """
        self.load( starName )
        
        defocusU = self.star['data_']['DefocusU']
        defocusV = self.star['data_']['DefocusV']
        finalResolution = self.star['data_']['FinalResolution']
        ctfFoM = self.star['data_']['CtfFigureOfMerit']
        
        defocusMean = 0.5 * defocusU + 0.5 * defocusV
        astig = np.abs( defocusU - defocusV )
        
        [hDefocus, cDefocus] = np.histogram( defocusMean,  
            bins=np.arange(np.min(defocusMean),np.max(defocusMean),500.0) )
        hDefocus = hDefocus.astype('float32')
        cDefocus = cDefocus[:-1] +500.0/2
        
        [hAstig, cAstig] = np.histogram( astig, 
            bins=np.arange(0, np.max(astig), 500.0) )
        hAstig = hAstig.astype('float32')
        cAstig = cAstig[:-1] +500.0/2
        
        [hFoM, cFoM] = np.histogram( ctfFoM,  
            bins=np.arange(0.0,np.max(ctfFoM),0.002) )
        hFoM = hFoM.astype('float32')
        cFoM = cFoM[:-1] +0.002/2.0
        
        [hRes, cRes] = np.histogram( finalResolution,  
            bins=np.arange(np.min(finalResolution),np.max(finalResolution),0.20) )
        hRes = hRes.astype('float32')
        cRes = cRes[:-1] +0.20/2.0
        
        plt.figure()
        plt.fill_between( cDefocus, hDefocus, np.zeros(len(hDefocus)), facecolor='steelblue', alpha=0.5 )
        plt.plot( [defocusThreshold, defocusThreshold], [0, np.max(hDefocus)], "--", color='firebrick' )
        plt.xlabel( "Defocus, $C_1 (\AA)$" )
        plt.ylabel( "Histogram counts" )
        
        plt.figure()
        plt.fill_between( cAstig, hAstig, np.zeros(len(hAstig)), facecolor='forestgreen', alpha=0.5 )
        plt.plot( [astigThreshold, astigThreshold], [0, np.max(hAstig)], "--", color='firebrick' )
        plt.xlabel( "Astigmatism, $A_1 (\AA)$" )
        plt.ylabel( "Histogram counts" )
        
        plt.figure()
        plt.fill_between( cFoM, hFoM, np.zeros(len(hFoM)), facecolor='darkorange', alpha=0.5 )
        plt.plot( [fomThreshold, fomThreshold], [0, np.max(hFoM)], "--", color='firebrick' )
        plt.xlabel( "Figure of Merit, $R^2$" )
        plt.ylabel( "Histogram counts" )
        
        plt.figure()
        plt.fill_between( cRes, hRes, np.zeros(len(hRes)), facecolor='purple', alpha=0.5 )
        plt.plot( [resThreshold, resThreshold], [0, np.max(hRes)], "--", color='firebrick' )
        plt.xlabel( "Fitted Resolution, $r (\AA)$" )
        plt.ylabel( "Histogram counts" )
        
        
        #keepIndices = np.ones( len(defocusU), dtype='bool' )
        keepIndices = ( ( defocusMean < defocusThreshold) & (astig < astigThreshold) &
                        (ctfFoM > fomThreshold ) & (finalResolution < resThreshold) )
        
        print( "KEEPING %d of %d micrographs" %(np.sum(keepIndices), defocusU.size) )
        
        for key in self.star['data_']:
            self.star['data_'][key] =  self.star['data_'][key][keepIndices]
            
        self.saveDataStar( outName )
        
    def __loadPar( self, parname ):
        """
        Frealign files normally have 16 columns, with any number of comment lines that start with 'C'
        """
        # Ergh, cannot have trailing comments with np.loadtxt?  
        self.parCol = [b"N", b"PSI", b"THETA", b"PHI", b"SHX", b"SHY", b"MAG", b"FILM", b"DF1", b"DF2", \
                     b"ANGAST", b"OCC", b"LogP", b"SIGMA", b"SCORE", b"CHANGE" ]
                     
        self.par = pandas.read_table( parname, engine='c', sep=' ', header=None, names =self.parCol, quotechar='C'  )
        #self.par.append( np.loadtxt( parname, comments=b'C' ) )
        # TODO: split into a dictionary?  
        # TODO: read comments as well
        # TODO: use pandas instead?
        #self.parCol = {b"N":0, b"PSI":1, b"THETA":2, b"PHI":3, b"SHX":4, b"SHY":5, b"MAG":6, b"FILM":7, b"DF1":8, b"DF2":9, 
        #             b"ANGAST":10, b"OCC":11, b"LogP":12, b"SIGMA":13, b"SCORE":14, b"CHANGE":15 }
        #self.parComments = np.loadtxt( parname, comments=b' ' )
        
               
    def __loadStar( self, starname ):
        with open( starname, 'rb' ) as starFile:
            starLines = starFile.readlines()
        
            # Remove any lines that are blank
            blankLines = [I for I, line in enumerate(starLines) if ( line == "\n" or line == " \n") ]
            for blank in sorted( blankLines, reverse=True ):
                del starLines[blank]
        
            # Top-level keys all start with data_
            headerTags = []; headerIndices = []
            for J, line in enumerate(starLines):
                if line.startswith( b"data_" ): # New headerTag
                    headerTags.append( line.strip() )
                    headerIndices.append( J )
            # for end-of-file
            headerIndices.append(-1)
            
            # Build dict keys
            for K, tag in enumerate( headerTags ):
                self.star[tag] = OrderedDict()
                # Read in _rln lines and assign them as dict keys
                
                lastHeaderIndex = 0
                foundLoop = False
                
                if headerIndices[K+1] == -1: #-1 is not end of the array for indexing
                    slicedLines = starLines[headerIndices[K]:]
                else:
                    slicedLines = starLines[headerIndices[K]:headerIndices[K+1]] 
                    
                for J, line in enumerate( slicedLines ):
                    if line.startswith( b"loop_" ):
                        foundLoop = True
                    elif line.startswith( b"_rln" ):
                        lastHeaderIndex = J
                        # Find all the keys that start with _rln, they are sub-dict keys
                        newKey = line.split()[0][4:]
                        try:
                            newValue = line.split()[1]
                            # If newValue starts with a #, strip it
                            newValue = newValue.lstrip( b'#' )
                        except:
                            # Some really old Relion star files don't have the column numbers, so assume it's ordered
                            newValue = J
                            
                        # Try to make newValue an int or float, or leave it as a string if that fails
                        try:
                            self.star[tag][newKey] = np.int( newValue )
                        except:
                            try: 
                                self.star[tag][newKey] = np.float( newValue )
                            except: # leave as a string
                                self.star[tag][newKey] = newValue
        
                # Now run again starting at lastHeaderIndex
                if foundLoop: 
                    # Need to check to make sure it's not an empty dict
                    if self.star[tag] == OrderedDict():
                        continue
                    
                    # Sometimes we have an empty line on the end.
                    for J in np.arange(len(slicedLines)-1,0,-1):
                        if bool( slicedLines[J].strip() ):
                            break
                        slicedLines = slicedLines[:J]

                    endIndex = len(slicedLines)
                    
                    
                        
                    # Reverse sub-dictionary so we can determine by which column goes to which key
                    lookup = dict( zip( self.star[tag].values(), self.star[tag].keys() ) )
                    print( "DEBUG: lookup = %s" % lookup )
                    # Pre-allocate, we can determine types later.   
                    itemCount = endIndex - lastHeaderIndex - 1
                    testSplit = slicedLines[lastHeaderIndex+1].split()
                    for K, test in enumerate( testSplit ):
                        self.star[tag][lookup[K+1]] = [None] * itemCount
        
                    # Loop through and parse items
                    for J, line in enumerate( slicedLines[lastHeaderIndex+1:endIndex] ):
                        for K, item in enumerate( line.split() ):
                            self.star[tag][lookup[K+1]][J] = item
                    pass
                
                    # Try to convert to int, then float, otherwise leave as a string
                    for key in self.star[tag].keys():
                        try:
                            self.star[tag][key] = np.asarray( self.star[tag][key], dtype='int' )
                        except:
                            try:
                                self.star[tag][key] = np.asarray( self.star[tag][key], dtype='float' )
                            except: 
                                self.star[tag][key] = np.asarray( self.star[tag][key] )
                                pass

    def __loadMRC( self, mrcname ):
        mrcimage, mrcheader  = ioMRC.readMRC( mrcname, pixelunits=u'nm' )
        self.mrc.append( mrcimage )
        self.mrc_header.append( mrcheader )
        
    def __loadBox( self, boxname ):
        self.box.append( np.loadtxt( boxname ) )

# End of relion class



