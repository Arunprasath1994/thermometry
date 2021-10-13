#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 19:01:47 2021

@author: arun
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 16:41:56 2021

@author: arun
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:34:35 2021

@author: arun
"""

''' winspec.py - read SPE files created by WinSpec with Princeton Instruments' cameras. '''

import ctypes, os
import struct
import numpy as np
import logging
log = logging.getLogger('winspec')

# Definitions of types
spe_byte = ctypes.c_ubyte
spe_word = ctypes.c_ushort
spe_dword = ctypes.c_uint

spe_char = ctypes.c_char # 1 byte
spe_short = ctypes.c_short # 2 bytes

# long is 4 bytes in the manual. It is 8 bytes on my machine
spe_long = ctypes.c_int # 4 bytes

spe_float = ctypes.c_float # 4 bytes
spe_double = ctypes.c_double # 8 bytes

class ROIinfo(ctypes.Structure):
    pass

class AxisCalibration(ctypes.Structure):
    pass

class Header(ctypes.Structure):
    pass

def print_offsets():
    ''' Print the attribute names, sizes and offsets in the C structure

    Assuming that the sizes are correct and add up to an offset of 4100 bytes,
    everything should add up correctly. This information was taken from the
    WinSpec 2.6 Spectroscopy Software User Manual version 2.6B, page 251.
    If this table doesn't add up, something changed in the definitions of the
    datatype widths. Fix this in winspec.structs file and let me know!
    '''

    import inspect, re

    A = Header()

    for i in [Header, AxisCalibration, ROIinfo]:
        fields = []

        print('\n{:30s}[{:4s}]\tsize'.format(repr(i), 'offs'))

        for name,obj in inspect.getmembers(i):
            if inspect.isdatadescriptor(obj) and not inspect.ismemberdescriptor(obj) \
                and not inspect.isgetsetdescriptor(obj):

                fields.append((name, obj))

        fields = sorted(fields, key=lambda x: x[1].offset)

        for name, obj in fields:
            print('{:30s}[{:4d}]\t{:4d}'.format(name, obj.size, obj.offset))


class SpeFile(object):
    ''' A file that represents the SPE file.
    All details written in the file are contained in the `header` structure. Data is
    accessed by using the `data` property.
    Once the object is created and data accessed, the file is NOT read again. Create
    a new object if you want to reread the file.
    '''

    # Map between header datatype field and numpy datatype
    _datatype_map = {0 : np.float32, 1 : np.int32, 2 : np.int16, 3 : np.uint16}

    def __init__(self, name):
        ''' Open file `name` to read the header.'''

        with open(name, mode='rb') as f:
            self.header = Header()
            self.path = os.path.realpath(name)
            self._data = None
            self._xaxis = None
            self._yaxis = None

            # Deprecated method, but FileIO apparently can't be used with numpy
            f.readinto(self.header)

        # set some useful properties
        self.reversed = True if self.header.geometric == 2 else False
        self.gain = self.header.gain

        if self.header.ADCtype == 8:
            self.adc = 'Low Noise'
        elif self.header.ADCtype == 9:
            self.adc = 'High Capacity'
        else:
            self.adc = 'Unknown'

        if self.header.ADCrate == 12:
            self.adc_rate = '2 MHz'
        elif self.header.ADCrate == 6:
            self.adc_rate = '100 KHz'
        else:
            self.adc_rate = 'Unknown'

        self.readout_time = self.header.ReadoutTime

    def _read(self):
        ''' Read the data segment of the file and create an appropriately-shaped numpy array
        Based on the header, the right datatype is selected and returned as a numpy array.  I took
        the convention that the frame index is the first, followed by the x,y coordinates.
        '''

        if self._data is not None:
            log.debug('using cached data')
            return self._data

        # In python 2.7, apparently file and FileIO cannot be used interchangably
        with open(self.path, mode='rb') as f:
            f.seek(4100) # Skip header (4100 bytes)

            _count = self.header.xdim * self.header.ydim * self.header.NumFrames

            self._data = np.fromfile(f, dtype=SpeFile._datatype_map[self.header.datatype], count=_count)

            # Also, apparently the ordering of the data corresponds to how it is stored by the shift register
            # Thus, it appears a little backwards...
            self._data = self._data.reshape((self.header.NumFrames, self.header.ydim, self.header.xdim))

            # Orient the structure so that it is indexed like [NumFrames][x, y]
            self._data = np.rollaxis(self._data, 2, 1)

            # flip data
            if all([self.reversed == True, self.adc == '100 KHz']):
                pass
            elif any([self.reversed == True, self.adc == '100 KHz']):
                self._data = self._data[:, ::-1, :]
                log.debug('flipped data because of nonstandard ADC setting ' +\
                        'or reversed setting')

            return self._data

    @property
    def xaxis(self):
        if self._xaxis is not None:
            log.debug('using cached xaxis')
            return self._xaxis

        px, py = self._make_axes()

        return px

    @property
    def yaxis(self):
        if self._yaxis is not None:
            log.debug('using cached yaxis')
            return self._yaxis

        px, py = self._make_axes()

        return py

    @property
    def xaxis_label(self):
        '''Read the x axis label
        '''
        return self.header.xcalibration.string.decode('ascii')

    @property
    def yaxis_label(self):
        '''Read the y axis label
        '''
        return self.header.ycalibration.string.decode('ascii')


    def _make_axes(self):
        '''Construct axes from calibration fields in header file
        '''
        xcalib = self.header.xcalibration
        ycalib = self.header.ycalibration

        xcalib_valid = struct.unpack('?', xcalib.calib_valid)

        if xcalib_valid:
            xcalib_order, = struct.unpack('>B', xcalib.polynom_order) # polynomial order
            px = xcalib.polynom_coeff[:xcalib_order+1]
            px = np.array(px[::-1]) # reverse coefficients to use numpy polyval
            pixels = np.arange(1, self.header.xdim + 1)
            px = np.polyval(px, pixels)
        else:
            px = np.arange(1, self.header.xdim + 1)

        ycalib_valid = struct.unpack('?', ycalib.calib_valid)

        if ycalib_valid:
            ycalib_order, = struct.unpack('>B', ycalib.polynom_order) # polynomial order
            py = ycalib.polynom_coeff[:ycalib_order+1]
            py = np.array(py[::-1]) # reverse coefficients to use numpy polyval
            pixels = np.arange(1, self.header.ydim + 1)
            py = np.polyval(py, pixels)
        else:
            py = np.arange(1, self.header.ydim + 1)

        self._xaxis = px
        self._yaxis = py

        return px, py


    ''' Data recorded in the file, returned as a numpy array.

    The convention for indexes is that the first index is the frame index, followed by x,y region of
    interest.
    '''
    data = property(fget=_read)

    def __str__(self):
        return 'SPE File \n\t{:d}x{:d} area, {:d} frames\n\tTaken on {:s}' \
                .format(self.header.xdim, self.header.ydim,
                        self.header.NumFrames, self.header.date.decode())

    def __repr__(self):
        return str(self)


# Lengths of arrays used in header
HDRNAMEMAX = 120
USERINFOMAX = 1000
COMMENTMAX = 80
LABELMAX = 16
FILEVERMAX = 16
DATEMAX = 10
ROIMAX = 10
TIMEMAX = 7

# Definitions of WinSpec structures

# Region of interest defs
ROIinfo._pack_ = 1
ROIinfo._fields_ = [
    ('startx', spe_word),
    ('endx', spe_word),
    ('groupx', spe_word),
    ('starty', spe_word),
    ('endy', spe_word),
    ('groupy', spe_word)]

# Calibration structure for X and Y axes
AxisCalibration._pack_ = 1
AxisCalibration._fields_ = [
    ('offset', spe_double),
    ('factor', spe_double),
    ('current_unit', spe_char),
    ('reserved1', spe_char),
    ('string', spe_char * 40),
    ('reserved2', spe_char * 40),
    ('calib_valid', spe_char),
    ('input_unit', spe_char),
    ('polynom_unit', spe_char),
    ('polynom_order', spe_char),
    ('calib_count', spe_char),
    ('pixel_position', spe_double * 10),
    ('calib_value', spe_double * 10),
    ('polynom_coeff', spe_double * 6),
    ('laser_position', spe_double),
    ('reserved3', spe_char),
    ('new_calib_flag', spe_byte),
    ('calib_label', spe_char * 81),
    ('expansion', spe_char * 87)]

# Full header definition
Header._pack_ = 1
Header._fields_ = [
    ('ControllerVersion', spe_short),
    ('LogicOutput', spe_short),
    ('AmpHiCapLowNoise', spe_word),
    ('xDimDet', spe_word),
    ('mode', spe_short),
    ('exp_sec', spe_float),
    ('VChipXdim', spe_short),
    ('VChipYdim', spe_short),
    ('yDimDet', spe_word),
    ('date', spe_char * DATEMAX),
    ('VirtualChipFlag', spe_short),
    ('Spare_1', spe_char * 2), # Unused data
    ('noscan', spe_short),
    ('DetTemperature', spe_float),
    ('DetType', spe_short),
    ('xdim', spe_word),
    ('stdiode', spe_short),
    ('DelayTime', spe_float),
    ('ShutterControl', spe_word),
    ('AbsorbLive', spe_short),
    ('AbsorbMode', spe_word),
    ('CanDoVirtualChipFlag', spe_short),
    ('ThresholdMinLive', spe_short),
    ('ThresholdMinVal', spe_float),
    ('ThresholdMaxLive', spe_short),
    ('ThresholdMaxVal', spe_float),
    ('SpecAutoSpectroMode', spe_short),
    ('SpecCenterWlNm', spe_float),
    ('SpecGlueFlag', spe_short),
    ('SpecGlueStartWlNm', spe_float),
    ('SpecGlueEndWlNm', spe_float),
    ('SpecGlueMinOvrlpNm', spe_float),
    ('SpecGlueFinalResNm', spe_float),
    ('PulserType', spe_short),
    ('CustomChipFlag', spe_short),
    ('XPrePixels', spe_short),
    ('XPostPixels', spe_short),
    ('YPrePixels', spe_short),
    ('YPostPixels', spe_short),
    ('asynen', spe_short),
    ('datatype', spe_short), # 0 - float, 1 - long, 2 - short, 3 - ushort
    ('PulserMode', spe_short),
    ('PulserOnChipAccums', spe_word),
    ('PulserRepeatExp', spe_dword),
    ('PulseRepWidth', spe_float),
    ('PulseRepDelay', spe_float),
    ('PulseSeqStartWidth', spe_float),
    ('PulseSeqEndWidth', spe_float),
    ('PulseSeqStartDelay', spe_float),
    ('PulseSeqEndDelay', spe_float),
    ('PulseSeqIncMode', spe_short),
    ('PImaxUsed', spe_short),
    ('PImaxMode', spe_short),
    ('PImaxGain', spe_short),
    ('BackGrndApplied', spe_short),
    ('PImax2nsBrdUsed', spe_short),
    ('minblk', spe_word),
    ('numminblk', spe_word),
    ('SpecMirrorLocation', spe_short * 2),
    ('SpecSlitLocation', spe_short * 4),
    ('CustomTimingFlag', spe_short),
    ('ExperimentTimeLocal', spe_char * TIMEMAX),
    ('ExperimentTimeUTC', spe_char * TIMEMAX),
    ('ExposUnits', spe_short),
    ('ADCoffset', spe_word),
    ('ADCrate', spe_word),
    ('ADCtype', spe_word),
    ('ADCresolution', spe_word),
    ('ADCbitAdjust', spe_word),
    ('gain', spe_word),
    ('Comments', spe_char * 5 * COMMENTMAX),
    ('geometric', spe_word), # x01 - rotate, x02 - reverse, x04 flip
    ('xlabel', spe_char * LABELMAX),
    ('cleans', spe_word),
    ('NumSkpPerCln', spe_word),
    ('SpecMirrorPos', spe_short * 2),
    ('SpecSlitPos', spe_float * 4),
    ('AutoCleansActive', spe_short),
    ('UseContCleansInst', spe_short),
    ('AbsorbStripNum', spe_short),
    ('SpecSlipPosUnits', spe_short),
    ('SpecGrooves', spe_float),
    ('srccmp', spe_short),
    ('ydim', spe_word),
    ('scramble', spe_short),
    ('ContinuousCleansFlag', spe_short),
    ('ExternalTriggerFlag', spe_short),
    ('lnoscan', spe_long), # Longs are 4 bytes
    ('lavgexp', spe_long), # 4 bytes
    ('ReadoutTime', spe_float),
    ('TriggeredModeFlag', spe_short),
    ('Spare_2', spe_char * 10),
    ('sw_version', spe_char * FILEVERMAX),
    ('type', spe_short),
    ('flatFieldApplied', spe_short),
    ('Spare_3', spe_char * 16),
    ('kin_trig_mode', spe_short),
    ('dlabel', spe_char * LABELMAX),
    ('Spare_4', spe_char * 436),
    ('PulseFileName', spe_char * HDRNAMEMAX),
    ('AbsorbFileName', spe_char * HDRNAMEMAX),
    ('NumExpRepeats', spe_dword),
    ('NumExpAccums', spe_dword),
    ('YT_Flag', spe_short),
    ('clkspd_us', spe_float),
    ('HWaccumFlag', spe_short),
    ('StoreSync', spe_short),
    ('BlemishApplied', spe_short),
    ('CosmicApplied', spe_short),
    ('CosmicType', spe_short),
    ('CosmicThreshold', spe_float),
    ('NumFrames', spe_long),
    ('MaxIntensity', spe_float),
    ('MinIntensity', spe_float),
    ('ylabel', spe_char * LABELMAX),
    ('ShutterType', spe_word),
    ('shutterComp', spe_float),
    ('readoutMode', spe_word),
    ('WindowSize', spe_word),
    ('clkspd', spe_word),
    ('interface_type', spe_word),
    ('NumROIsInExperiment', spe_short),
    ('Spare_5', spe_char * 16),
    ('controllerNum', spe_word),
    ('SWmade', spe_word),
    ('NumROI', spe_short),
    ('ROIinfblk', ROIinfo * ROIMAX),
    ('FlatField', spe_char * HDRNAMEMAX),
    ('background', spe_char * HDRNAMEMAX),
    ('blemish', spe_char * HDRNAMEMAX),
    ('file_header_ver', spe_float),
    ('YT_Info', spe_char * 1000),
    ('WinView_id', spe_long),
    ('xcalibration', AxisCalibration),
    ('ycalibration', AxisCalibration),
    ('Istring', spe_char * 40),
    ('Spare_6', spe_char * 25),
    ('SpecType', spe_byte),
    ('SpecModel', spe_byte),
    ('PulseBurstUsed', spe_byte),
    ('PulseBurstCount', spe_dword),
    ('PulseBurstPeriod', spe_double),
    ('PulseBracketUsed', spe_byte),
    ('PulseBracketType', spe_byte),
    ('PulseTimeConstFast', spe_double),
    ('PulseAmplitudeFast', spe_double),
    ('PulseTimeConstSlow', spe_double),
    ('PulseAmplitudeSlow', spe_double),
    ('AnalogGain', spe_short),
    ('AvGainUsed', spe_short),
    ('AvGain', spe_short),
    ('lastvalue', spe_short)]

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#




























"PIV code"

"""
Function to make correlation between two images and return a
correlation matrix along with the top-left corner of the best
matched template.
Arguments : template, search area

Variables and description :
   
img     :Bigger image where the template is to be matched
img2    :Copy of the bigger image
methods :Specification of the type of correlation to be used.
         Available methods are
         'cv2.TM_CCOEFF',
         'cv2.TM_CCOEFF_NORMED',
         'cv2.TM_CCORR',
         'cv2.TM_CCORR_NORMED',
         'cv2.TM_SQDIFF',
         'cv2.TM_SQDIFF_NORMED'
         See https://docs.opencv.org/3.0-beta/doc/py_tutorials/
         py_imgproc/py_template_matching/py_template_matching.html
        
template :The window to be matched in the bigger search area
res1     :The corelation matrix
min_val  :The value of the least correlated value , used in case of
          Squared difference methods.
max_val  :The value with the maximum correlation value used in all
          cases other than squared difference
min_loc  :Best matched location with respect to squared difference method
max_loc  :Best matched location with respect to cross corelations
  
"""


def makecorr(imga,imgb):

    img = imga
    img = np.array(img,dtype='float32')
    img2 = img.copy()
    template = imgb
    template = np.array(template,dtype='float32')
   
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCORR_NORMED']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
           
#        center = (top_left[0], top_left[1] )
       
    return res,top_left

"""
Function to make Sub-pixel estimation : Gaussian fit
arguments : correlation matrix, location of maximum correlation

peakj, peaki :coordinates of location of the maximum correlation
                  in correlation matrix
c              :  corelation coefficient at maximum location
cl,cr,cd,cu    :  corelation coefficients at the left, right, down and
                  upper cells adjacent to the maximum value
subp           :  x,y coordinates of the best matched location after subpixel
                  estimation 
                 
                
"""

   
def subpixel(mat,location):
       
    peakj, peaki = location
 
    # the peak and its neighbours: left, right, down, up
    c  = mat[peaki,   peakj]
    cl = mat[peaki - 1, peakj]
    cr = mat[peaki + 1, peakj]
    cd = mat[peaki,   peakj - 1]
    cu = mat[peaki,   peakj + 1]
    if c>0.8:
        subp =(peaki + ((np.log10(cl) - np.log10(cr)) / (2 * np.log10(cl) -
                         4 * np.log10(c) + 2 * np.log10(cr))),
               peakj + ((np.log10(cd) - np.log10(cu)) / (2 * np.log10(cd) -
                         4 * np.log10(c) + 2 * np.log10(cu))))
    else:
        subp=location
        
               
    return subp      

 





      
#-------------------------------------------------------------------------------------------------------------
def piv(img_left,img_right):
    start1=time.time()
    """Read the images before and after movement
    img1 = Image at time t=t0
    img2 = Image at time t=t0+dt
    """
                 
    #img1=cv2.imread("initial_image1.jpg",0)
    #img2=cv2.imread("displaced1.jpg",0)
    
    img1=img_left
    img2=img_right
    
    """
    w,h = width and height of the image in pixels"""
    w, h = img1.shape[::-1]
    """specification of the window size
    
    d2=     size of the template to be matched in pixels . In this case , d2=16
            lation was unable to be performed
    U=      Velocity in X direction  ( in mm/second)
    V=      Velocity in Y direction  ( in mm/second)pixels,template is 16x16 pixels
    d=      number of pixels extra surrounding the template, so as to have bigger
            search area
            in this case, we have 8 pixels to the left, right , top and bottom ,
            which makes the search area to be 32 x32
    n=      width of the image / side length of the template in pixels
    m=      height of the image / side length of the template in pixels
    k=      Counting variable
    num=    array length (usually nxm)
    windowx=Length of the domain in the x direction in mm
    windowy=Length of the domain in the y direction in mm
    dt=     Time between the two images
    factorx=Factor in x direction to be multiplied to get displacement in mm
    factory=Factor in y direction to be multiplied to get displacement in mm
    xb=     array with the x coordinates of all matched locations
    yb=     array with the y coordinates of all matched locations
    xt=     True x coordinate of the template
    yt=     True x coordinate of the template
    x1=     Left limit of the search area
    x2=     Left limit of the template
    x3=     Right limit of the template
    x4=     Right limit of the search area
    y1=     top limit of the search area
    y2=     top limit of the template
    y3=     Bottom limit of the template
    y4=     Bottom limit of the template
    img3=   Template cut out of the image at time t=t0
    img4=   Search area cut out of the image at time t=t+dt
    xc=     Location of the top left corner of the best match
    res1=   correlation coefficient matrix
    cent=   location of best match after subpixel estimation
    z=      Absolute velocity
    error=  Stores the locations where the coorelation was unable to be performed
    U=      Velocity in X direction  ( in mm/second)
    V=      Velocity in Y direction  ( in mm/second)
    """
       
       
    d2= 64
    d=  32
    n=  int(w/(2*d)-1)
    m=  int(h/(2*d)-1)
    k=  0
    variable1=0
    num=n*m
    windowx=100
    windowy=100
    dt=1/5
#    factorx=windowx/(w*dt)
#    factory=windowy/(h*dt)
    xb=np.zeros([num,1])
    yb=np.zeros([num,1]) #zeroes because ...
    xt=np.zeros([num,1])
    yt=np.zeros([num,1])
    error=np.zeros([num,1])
    
    for i in range (0,m):
        
        for j in range(0,n):          # the image is of dimensions 1280x1024.
            x=d2+i*d2  
            y=d2+j*d2
            x1=x-d2
            x2=x-d
            x3=x+d
            x4=x+d2
            y1=y-d2
            y2=y-d
            y3=y+d
            y4=y+d2
        
           
    ##imgB=img1[y1:y2, x1:x2] #template to be matched in image 2
           
    # We use Correlation function to obtain the location of the matched area.   
    # we need to crop the images from the main image according to the locations
    # specified above.Image A is the template and Image B is the search area which
    # is larger by a width of d/2 on all sides of the template imageA  
     
            img3=img1[x2:x3,y2:y3] #template to be matched in image 2
            img4=img2[x1:x4,y1:y4] #Bigger search area to find the matching image
    #         
            res1,xc=makecorr(img4,img3)
    
            if xc[0] >= d2 or xc[1] >= d2:
                xb[k],yb[k]=x,y
                error[variable1]=k
                variable1+=1
            else :
                cent=subpixel(res1,xc)
                xb[k],yb[k]=cent[1]+x1+d,cent[0]+y1+d
                xt[k],yt[k]=x,y
               
                k=k+1
               
    X, Y, U, V = yt,xt,(xb-xt),(yb-yt)
    z1=U*1
    #z=np.sqrt(U*U+V*V)
    #z1 = z1[np.logical_not(np.isnan(z1))]
    
    z2=V*1
    #z=np.sqrt(U*U+V*V)
    #z2 = z2[np.logical_not(np.isnan(z2))]
    
    z=z1*z1+z2*z2
    z=np.sqrt(z)
    z = z[np.logical_not(np.isnan(z))]
         
    
    
#    new1 =np.reshape(U,(m,n))
#    new2 =np.reshape(V,(m,n))
#    
#    diffx=np.diff(new1)
#    diffy=np.diff(new1,axis=0)
#    grad1=np.sum(diffx)
#    grad2=np.sum(diffy)
    
    #new=new1[x2:x3,y2:y3]
    #div=np.gradient(new)
    #div1=np.diff(new)
    #div2=np.sum(div1)
    
    """median test
    
    The median test is necessary to remove outliers .
    Since, we have just a 1-D array of values for U,V,Z we need to
    use the values somehow to find adjacent values.
    t= The row on the top , (i-1)
    c= The row same as the center i
    r= The row below i+1
    tr=top-right
    tc=top center
    tl=top left
    cr=center right
    cc=center center
    cl=center left
    br=bottom right
    bc=bottom center
    bl=bottom left
    
    In a matrix as
    1 2 3
    4 5 6
    7 8 9
    
    if 5 is cc,
     
    tl=1 , tc=2 tr=3
    cl=4   cr=6
    ll=7   lc=8 lr=9
    
    sm= median value of all the cells surrounding (i,j)th cell
    r = (value at (i,j) - sm)/sigma
    we consider sigma = 1
    
    " Particle image velocimetry - R.J Adrian , page 422 "
    """
    #sigma=1
    #for i in range (1,m-2):
    #     t=(i-1)*(n+1)
    #     c=i*(n+1)
    #     b=(i+1)*(n+1)
    #     for j in range(1,n-2): 
    #         tr=t+j+1
    #         tc=t+j
    #         tl=t+j-1
    #         cr=c+j+1
    #         cc=c+j
    #         cl=c+j-1
    #         br=b+j+1
    #         bc=b+j
    #         bl=b+j-1
    #        
    #         sm=(z[tr]+z[tc]+z[tl]+z[cl]+z[cr]+z[bl]+z[bc]+z[br])/8
    #         r=np.abs((z[cc]-sm))/(sm*sigma)
    #         if r>1.5 or r<0.5:
    #             
    #             U[cc]=(U[cl]+U[cr])/2 #interpolating values for U,V
    #             V[cc]=(V[cl]+V[cr])/2
    #           
    
    
    
    """ The final task is to present the vectors in the a graph"""  
    print("reduce left picture width by %f pixels and reduce height by %f" %(np.mean(U),np.mean(V)))
    norm = matplotlib.colors.Normalize()
    norm.autoscale(z)
    cm = matplotlib.cm.rainbow
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlabel('distance in x (pixels)')
    ax.set_ylabel('distance in y (pixels)')
    ax.set_xlim([0, w]) #x,y limits
    ax.set_ylim([h,0])
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    plt.quiver(X,Y,U,V,angles='xy', scale_units='xy',headwidth=8,width=0.003
              ) #properties of the plot
    plt.xlabel='plot'
    plt.colorbar(sm)
    plt.savefig('vectors.jpg',dpi=1000) #Title and location of image to save
    plt.show()
    
    
    X1=X*windowx/w
    Y1=Y*windowy/h
    
    
    
    
    
    print("--- %s seconds ---" % (time.time() - start1))
    return np.mean(U),np.mean(V)
    
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------













"Save Left and Right images"
def imp_img(T):
    for a in (T):
        rep='/media/arun/USB DISK/19-05/temp/' 
        img=SpeFile(rep+'%s.SPE' %a)._read()





"choose pictures with illumination"

def load_image1(repBG,BG,rep,T1,x1,x2,y1,y2,l1,l2,b1,b2,length,width):

    image_mean_BG=np.zeros((x2-x1,y2-y1),float)
    imgBG=SpeFile(repBG+BG)._read()
    for i in range (len (imgBG)): 
        
        image_mean_BG=image_mean_BG + imgBG[i,x1:x2,y1:y2]

    image_mean_BG=image_mean_BG/len(imgBG)
    print('10')
    count=0
     
    img_moyL=np.zeros(len(T1),int)
    img_moyR=np.zeros(len(T1),int)
    image_meanL=np.zeros((len(T1),length,width),int)
    image_meanR=np.zeros((len(T1),length,width),int)  
    for a in (T1):
        
        kkL=np.zeros((len(T1),length,width),int)
        kkR=np.zeros((len(T1),length,width),int) 
        
        count1=0  
        print(count1)
        img=SpeFile(rep+'%s.SPE' %a)._read()-image_mean_BG
        img= ndimage.rotate(img, 90,axes=(2, 1))
        img_left=img[:,l1:l1+length,b1:b1+width]
        img_right=img[:,l2:l2+length,b2:b2+width]  

        for i in range(len (img)):   
            print(i)
            
#            plt.imsave('/media/arun/USB DISK/images/imgL_T%s_%d.png' %(a,i),img_left[i])
#            plt.imsave('/media/arun/USB DISK/images/imgR_T%s_%d.png' %(a,i),img_right[i])
            if np.mean(img_left[i])>50:
                img_moyL[count]=np.mean(img_left[i])+img_moyL[count]
                img_moyR[count]=np.mean(img_right[i])+img_moyR[count]
                count1=count1+1
                print('2')
            
      
                       
            for l in range(0,length):
                for m in range(0,width):
                    if img_left[i,l,m]>50:
                        image_meanL[count,l,m]=image_meanL[count,l,m] + img_left[i,l,m]
                        kkL[count,l,m]=kkL[count,l,m]+1
                        

                    if img_right[i,l,m]>50:
                        image_meanR[count,l,m]=image_meanR[count,l,m] + img_right[i,l,m]
                        kkR[count,l,m]=kkR[count,l,m]+1
                        
            
        
        img_moyL[count]=img_moyL[count]/count1            
        img_moyR[count]=img_moyR[count]/count1      
        count1=0  
        for l in range(0,length):
            for m in range(0,width):            
                if kkL[count,l,m]>0 :      
                    image_meanL[count,l,m]=image_meanL[count,l,m]/kkL[count,l,m]      
        for l in range(0,length):
            for m in range(0,width):          
                if kkR[count,l,m]>0 :      
                    image_meanR[count,l,m]=image_meanR[count,l,m]/kkR[count,l,m]
        # plt.imsave('/media/arun/USB DISK/images/NEWimgL_T%s_mean.png' %(a),image_meanL[count])
        # plt.imsave('/media/arun/USB DISK/images/NEWimgR_T%s_mean.png' %(a),image_meanR[count])  
          
        
    #     plt.plot(image_meanL[count,:,50],'r.')
    #     plt.plot(image_meanL[count,:,100],'y.')
    #     plt.plot(image_meanL[count,:,150],'g.')
    #     plt.plot(image_meanL[count,:,300],'b.')
    #     ax = plt.gca()
    #     ax.set_xlabel('Pixel location along vertical direction')
    #     ax.set_ylabel('Intensity')
    #     plt.title('Left Image')
    #     plt.savefig('/media/arun/USB DISK/images/NEWimgR_T%s_pixInt_horizontal.pdf' %(a))
    #     plt.show()
        
    #     plt.plot(image_meanR[count,:,50],'r.')
    #     plt.plot(image_meanR[count,:,100],'y.')
    #     plt.plot(image_meanR[count,:,150],'g.')
    #     plt.plot(image_meanR[count,:,350],'b.')
    #     ax = plt.gca()
    #     ax.set_xlabel('Pixel location along vertical direction')
    #     ax.set_ylabel('Intensity')
    #     plt.title('Right Image')
    #     plt.savefig('/media/arun/USB DISK/images/NEWimgR_T%s_pixInt_horizontal.pdf' %(a))
    #     plt.show()
        
    #     plt.plot(image_meanR[count,50,:],'r.')
    #     plt.plot(image_meanR[count,100,:],'y.')
    #     plt.plot(image_meanR[count,150,:],'g.')
    #     plt.plot(image_meanR[count,350,:],'b.')
    #     ax = plt.gca()
    #     ax.set_xlabel('Pixel location along horizontal direction')
    #     ax.set_ylabel('Intensity')
    #     plt.title('Right Image')
    #     plt.savefig('/media/arun/USB DISK/images/NEWimgR_T%s_pixInt_horizontal.pdf' %(a))
    #     plt.show()
        count=count+1
    plt.plot(T1,img_moyL,'r.')
    plt.plot(T1,img_moyR,'b.')
    plt.show()    
            
    return image_meanL,image_meanR,img_moyL,img_moyR





def load_image(repBG,BG,rep,T,x1,x2,y1,y2,l1,l2,b1,b2,length,width):

    image_mean_BG=np.zeros((x2-x1,y2-y1),float)
    imgBG=SpeFile(repBG+BG)._read()
    for i in range (len (imgBG)): 
        
        image_mean_BG=image_mean_BG + imgBG[i,x1:x2,y1:y2]
        
    image_mean_BG=image_mean_BG/len(imgBG)
    print('10')
    count=0
     
    img_moyL=np.zeros(len(T),int)
    img_moyR=np.zeros(len(T),int)
    image_meanL=np.zeros((len(T),length,width),int)
    image_meanR=np.zeros((len(T),length,width),int)  
    for a in (T):
        kkL=np.zeros((len(T),length,width),int)
        kkR=np.zeros((len(T),length,width),int) 
        
        count1=0  
        
        img=SpeFile(rep+'%s.SPE' %a)._read()-image_mean_BG
        for i in range(len(img)):
            img[i]=cv2.GaussianBlur(img[i,:,:],(5,5),0)
        plt.imshow(img[5])
        plt.clf()
        img= ndimage.rotate(img, 90,axes=(2, 1))
        

        img_left=img[:,l1:l1+length,b1:b1+width]
        img_right=img[:,l2:l2+length,b2:b2+width]  

        for i in range(len (img)):   

            
            # plt.imsave('/media/arun/USB DISK/images/NEWimgL_T%s_%d.png' %(a,i),img_left[i])
            # plt.imsave('/media/arun/USB DISK/images/NEWimgR_T%s_%d.png' %(a,i),img_right[i])
            if np.mean(img_left[i])>50:
                img_moyL[count]=np.mean(img_left[i])+img_moyL[count]
                img_moyR[count]=np.mean(img_right[i])+img_moyR[count]
                count1=count1+1
#                print('2')
            
      
                       
            for l in range(0,length):
                for m in range(0,width):
                    if img_left[i,l,m]>50:
                        image_meanL[count,l,m]=image_meanL[count,l,m] + img_left[i,l,m]
                        kkL[count,l,m]=kkL[count,l,m]+1
                        

                    if img_right[i,l,m]>50:
                        image_meanR[count,l,m]=image_meanR[count,l,m] + img_right[i,l,m]
                        kkR[count,l,m]=kkR[count,l,m]+1
                        

        
        img_moyL[count]=img_moyL[count]/count1            
        img_moyR[count]=img_moyR[count]/count1      
        count1=0  
        for l in range(0,length):
            for m in range(0,width):            
                if kkL[count,l,m]>0 :      
                    image_meanL[count,l,m]=image_meanL[count,l,m]/kkL[count,l,m]      
        for l in range(0,length):
            for m in range(0,width):          
                if kkR[count,l,m]>0 :      
                    image_meanR[count,l,m]=image_meanR[count,l,m]/kkR[count,l,m]
        # plt.imsave('/media/arun/USB DISK/images/NEWimgL_T%s_mean.png' %(a),image_meanL[count])
        # plt.imsave('/media/arun/USB DISK/images/NEWimgR_T%s_mean.png' %(a),image_meanR[count])  
          
        
        # plt.plot(image_meanL[count,:,50],'r.')
        # plt.plot(image_meanL[count,:,100],'y.')
        # plt.plot(image_meanL[count,:,150],'g.')
        # plt.plot(image_meanL[count,:,300],'b.')
        # ax = plt.gca()
        # ax.set_xlabel('Pixel location along vertical direction')
        # ax.set_ylabel('Intensity')
        # plt.title('Left Image')
        # plt.savefig('/media/arun/USB DISK/images/NEWimgR_T%s_pixInt_horizontal.pdf' %(a))
        # plt.show()
    
        # plt.plot(image_meanR[count,:,50],'r.')
        # plt.plot(image_meanR[count,:,100],'y.')
        # plt.plot(image_meanR[count,:,150],'g.')
        # plt.plot(image_meanR[count,:,350],'b.')
        # ax = plt.gca()
        # ax.set_xlabel('Pixel location along vertical direction')
        # ax.set_ylabel('Intensity')
        # plt.title('Right Image')
        # # plt.savefig('/media/arun/USB DISK/images/NEWimgR_T%s_pixInt_horizontal.pdf' %(a))
        # plt.show()
        
        # plt.plot(image_meanR[count,50,:],'r.')
        # plt.plot(image_meanR[count,100,:],'y.')
        # plt.plot(image_meanR[count,150,:],'g.')
        # plt.plot(image_meanR[count,350,:],'b.')
        # ax = plt.gca()
        # ax.set_xlabel('Pixel location along horizontal direction')
        # ax.set_ylabel('Intensity')
        # plt.title('Right Image')
        # # plt.savefig('/media/arun/USB DISK/images/NEWimgR_T%s_pixInt_horizontal.pdf' %(a))
        # plt.show()
        count=count+1
    plt.plot(T,img_moyL,'r.')
    plt.plot(T,img_moyR,'b.')
    plt.show()    
            
    return image_meanL,image_meanR,img_moyL,img_moyR






#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


'''Bg subtraction, Mean of the mean images, Mean of every single image for each
   temperature and StD, Laser fluence absorption'''
def compute(imgL,imgR,length,width,L,B,T,ii):
    plt.clf()
    mean_div=np.zeros(len(T),float)
    mean_R=np.zeros(len(T),float)
    mean_L=np.zeros(len(T),float)
    imagediv=np.zeros((len(T),length,width),float)
    pix=8
    n1=int(length/pix)
    n2=int(width/pix)
    div=np.zeros((len(T),n1,n2),float)
    divT=np.zeros((len(T),n1,n2),float)
    for c in range(len(T)):
        imgLi=imgL[c,L:L+length,B:B+width]/imgL[ii,L:L+length,B:B+width]
        imgRi=imgR[c,L:L+length,B:B+width]/imgR[ii,L:L+length,B:B+width]
       
        levels1 = np.linspace(0.8,1.5,16)
        mean_L[c]=np.mean(imgLi)
        mean_R[c]=np.mean(imgRi)
        mean_div[c]=mean_L[c]/mean_R[c]

        
        imagediv[c]=imgLi/imgRi
       
        
        for i in range(n1):
            for j in range(n2):
                l=(pix*i)
                m=(pix*(i+1)-1)
                n=(pix*j)
                o=(pix*(j+1)-1)
                div[c,i,j]=np.mean(imgLi[l:m,n:o])/np.mean(imgRi[l:m,n:o])
        img66=plt.contourf(div[c],cmap='bwr',levels=levels1)
        plt.colorbar(img66, label='Intensity for %s' %T[c]) 
        plt.axis('equal')
    #    plt.imsave('/media/arun/USB DISK/images/imgR_T%s_mean.png' %(a),image_meanR[count]) 
        plt.imsave('/media/arun/DATA2/Phosphor thermometry/25august/DIV_T%s.png' %(T[c]),div[c])
        plt.show()
        plt.clf()
        print('standard deviation is %f' %np.std(div))
        print(mean_div[c])
        
        # plt.contourf(imagediv[c],levels = np.linspace(0.5,1.5,12))
        plt.show()
    plt.plot(T,mean_L,'r.')
    plt.plot(T,mean_R,'b.')
    plt.show()
    plt.plot(Ta,mean_div,'ro')
    # plt.plot(Tb,mean_div,'bo')
    plt.show()
    
    
    P=np.zeros((n1,n2,3),float)
    for i in range(n1):
        for j in range(n2):
            P[i,j,:]=np.polyfit(div[0:len(T),i,j],T,2)  
            
    
    


    # z=np.poly1d(P)
    # print(z(1.2))
    levels2 = np.linspace(20,55,32)

    for c in range(len(T)):
            for i in range(n1):
                for j in range(n2):
                    divT[c,i,j]=div[c,i,j]*div[c,i,j]*P[i,j,0]+div[c,i,j]*P[i,j,1]+P[i,j,2]
            imgT=plt.contourf(divT[c],cmap='bwr',levels=levels2)
            plt.colorbar(imgT, label='Intensity for %s' %T[c]) 
            plt.axis('equal')
        #    plt.imsave('/media/arun/USB DISK/images/imgR_T%s_mean.png' %(a),image_meanR[count]) 
            plt.imsave('/media/arun/DATA2/Phosphor thermometry/25august/DIV_Tc%s.png' %(T[c]),divT[c])
            plt.show()
            plt.clf()
    # print('standard deviation is %f' %np.std(div))
    # print(mean_div[c])
    
    return mean_div,imagediv,divT,P
    
    
    
    
    
    
    

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------






def computeRB(imgL,imgR,length,width,L,B,T,ii,IL,IR,P):
    pix=8
    n1=int(length/pix)
    n2=int(width/pix)
    div=np.zeros((len(IL),n1,n2),float)
    divT=np.zeros((len(IL),n1,n2),float)


    for c in range(len(IL)):
        imgLi=IL[c,L:L+length,B:B+width]
        imgRi=IR[c,L:L+length,B:B+width]
        
        
        for i in range(n1):
                for j in range(n2):
                    l=(pix*i)
                    m=(pix*(i+1)-1)
                    n=(pix*j)
                    o=(pix*(j+1)-1)
                    div[c,i,j]=np.mean(imgLi[l:m,n:o])/np.mean(imgRi[l:m,n:o])
    
    
    levels2 = np.linspace(15,30,32)
    for c in range(len(IL)):
        for i in range(n1):
            for j in range(n2):
                divT[c,i,j]=div[c,i,j]*div[c,i,j]*P[i,j,0]+div[c,i,j]*P[i,j,1]+P[i,j,2]
        imgT=plt.contourf(divT[c],cmap='bwr',levels=levels2)
        plt.colorbar(imgT, label='Intensity for %s' %c) 
        plt.axis('equal')
    #    plt.imsave('/media/arun/USB DISK/images/imgR_T%s_mean.png' %(a),image_meanR[count]) 
        # plt.imsave('/media/arun/DATA2/Phosphor thermometry/25august/DIV_Trbc%s.png' %(c),imgT)
        plt.show()
        plt.clf()
    
    return divT
    
    

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------










"Main program" 


import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.colors
from  PIL import Image
import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.optimize import curve_fit
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import numba
from numba import jit
from scipy import ndimage
from numba import njit, prange

x1=0
x2=1024
y1=0
y2=1024
list_im_mean=[]
S_laser=[]



"check for PIV"



# l1=280+150
# b1=115+150
# l2=284+150
# b2=572+150
# width=150
# length=150


l1=400
b1=100
l2=255+150
b2=558
width=480-100
length=300


print(l1,l2,b1,b2)


rep='/media/arun/DATA2/Phosphor thermometry/6september/' 
img=SpeFile(rep+'sansfiltres-60g.SPE')._read()
img= ndimage.rotate(img, 90,axes=(2, 1))
for i in range(len(img)):
     img[i]=cv2.GaussianBlur(img[i,:,:],(5,5),0)


fig,ax=plt.subplots()
ax.imshow(img[5])
rect=Rectangle((b1,l1),width,length,edgecolor='r',facecolor='none')
rect1=Rectangle((b2,l2),width,length,edgecolor='r',facecolor='none')
ax.add_patch(rect)
ax.add_patch(rect1)
plt.show()
plt.clf()

im_left=img[20,l1:l1+length,b1:b1+width]
im_right=img[20,l2:l2+length,b2:b2+width]

b,l=(piv(im_left,im_right))
l=int(round(l))
b=int(round(b))

l2=l2+l
b2=b2+b
print(l1,l2,b1,b2)
img_left=img[20,l1:l1+length,b1:b1+width]
img_right=img[20,l2:l2+length,b2:b2+width]
b,l=piv(img_left,img_right)

l=int(round(l))
b=int(round(b))

l2=l2+l
b2=b2+b
print(l1,l2,b1,b2)

img_left=img[20,l1:l1+length,b1:b1+width]
img_right=img[20,l2:l2+length,b2:b2+width]
b,l=piv(img_left,img_right)

l=int(round(l))
b=int(round(b))

l2=l2+l
b2=b2+b
print(l1,l2,b1,b2)





# im_left=img[7,l1:l1+length,b1:b1+width]
# im_right=img[7,l2:l2+length,b2:b2+width]

# plt.contourf(img[20])
# plt.axis('scaled')
# plt.show()
# plt.clf()
# plt.contourf(img_left)
# plt.axis('scaled')
# plt.show()
# plt.clf()
# plt.contourf(img_right)
# plt.axis('scaled')
# plt.show()
# plt.clf()



####################################################################################################################################    


" import images of real pictures " 
T=[20.5,22.5,25.5,30.2,30.5,35,40.4,45,49.7]
Ta=np.asarray(T)+273

T1=[60]




#Background images

BG='bg_180g.SPE'
rep='/media/arun/DATA2/Phosphor thermometry/25august/' 
repBG='/media/arun/DATA2/Phosphor thermometry/23august/23august/' 
image_meanL,image_meanR,img_mean_moyL,img_mean_moyR=load_image(repBG,BG,rep,T,x1,x2,y1,y2,l1,l2,b1,b2,length,width)

image_meanL1,image_meanR1,img_mean_moyL1,img_mean_moyR1=load_image(repBG,BG,rep,T1,x1,x2,y1,y2,l1,l2,b1,b2,length,width)



i1=0
L=0
B=0
length1=length
width1=width


# image_meanL_1=image_meanL
# image_meanR_1=image_meanR
# # image_meanL11=image_meanL1
# # image_meanR11=image_meanR1

   
# ####################################################################################################################################    


# matrix1=image_meanL[0]/np.mean(image_meanL[0])
# matrix2=image_meanR[0]/np.mean(image_meanR[0])
# # matrix3=image_meanL1[11]/np.mean(image_meanL1[11])
# # matrix4=image_meanR1[11]/np.mean(image_meanR1[11])


# for i in range(len(T)):
#     image_meanL[i]=(image_meanL[i]/matrix1)
#     image_meanR[i]=(image_meanR[i]/matrix2)
    
# # for i in range(len(T1)):
# #    image_meanL1[i]=(image_meanL1[i]/matrix3)
# #    image_meanR1[i]=(image_meanR1[i]/matrix4)    
    
# #image_mean=cv2.GaussianBlur(image_mean,(5,5),1)
# #list_im_mean.append(image_mean)
# #kk=np.zeros((x2,y2),int)

pix=8
n1=int(length/pix)
n2=int(width/pix)
i1=0
# # width1=721-366-10
divT=np.zeros((len(T),n1,n2),float)
meandiv=np.zeros(len(T),float)
imagediv=np.zeros((len(T),length1,width1),float)
P=np.zeros((n1,n2,3),float)
meandiv,imagediv,divT,P=compute(image_meanL,image_meanR,length1,width1,L,B,T,i1)  



# i2=1
# divT1=np.zeros((len(img),n1,n2),float)
# meandiv1=np.zeros(len(T),float)
# imagediv1=np.zeros((len(T),length1,width1),float)


for i in range(3):
    plt.hist(np.ravel(divT[i]),bins=100)


# image_mean_BG=np.zeros((x2-x1,y2-y1),float)
# imgBG=SpeFile(repBG+BG)._read()
# img=SpeFile(rep+'60.SPE')._read()-image_mean_BG
# img= ndimage.rotate(img, 90,axes=(2, 1))

# for i in range(len(img)):
#     img[i]=cv2.GaussianBlur(img[i,:,:],(23,23),0)
    
# img_left=img[:,l1:l1+length,b1:b1+width]
# img_right=img[:,l2:l2+length,b2:b2+width]  

        
# # img_left=img_left/matrix1
# # img_right=img_right/matrix2





matrix1=image_meanL[0]/np.mean(image_meanL[0])
matrix2=image_meanR[0]/np.mean(image_meanR[0])
# # matrix3=image_meanL1[11]/np.mean(image_meanL1[11])
# # matrix4=image_meanR1[11]/np.mean(image_meanR1[11])


for i in range(len(T)):
    image_meanL[i]=(image_meanL[i]/matrix1)
    image_meanR[i]=(image_meanR[i]/matrix2)
    
# # for i in range(len(T1)):
# #    image_meanL1[i]=(image_meanL1[i]/matrix3)
# #    image_meanR1[i]=(image_meanR1[i]/matrix4)    
    
# #image_mean=cv2.GaussianBlur(image_mean,(5,5),1)
# #list_im_mean.append(image_mean)
# #kk=np.zeros((x2,y2),int)

pix=8
n1=int(length/pix)
n2=int(width/pix)

# # width1=721-366-10
divT=np.zeros((len(T),n1,n2),float)
meandiv=np.zeros(len(T),float)
imagediv=np.zeros((len(T),length1,width1),float)
meandiv,imagediv,divT,P=compute(image_meanL,image_meanR,length1,width1,L,B,T,i1)  

i2=1
divT1=np.zeros((len(img),n1,n2),float)
meandiv1=np.zeros(len(T),float)
imagediv1=np.zeros((len(T),length1,width1),float)



image_mean_BG=np.zeros((x2-x1,y2-y1),float)
imgBG=SpeFile(repBG+BG)._read()
img=SpeFile(rep+'60.SPE')._read()-image_mean_BG
img= ndimage.rotate(img, 90,axes=(2, 1))

for i in range(len(img)):
    img[i]=cv2.GaussianBlur(img[i,:,:],(19,19),0)
    
img_left=img[:,l1:l1+length,b1:b1+width]
img_right=img[:,l2:l2+length,b2:b2+width]  

        
img_left=img_left/matrix1
img_right=img_right/matrix2


meandiv1,divT1=computeRB(image_meanL,image_meanR,length1,width1,L,B,T,i2,img_left,img_right,P)  






i1=0
L=0
B=0
length1=length
width1=width
divT1=computeRB(image_meanL1,image_meanR1,length1,width1,L,B,T1,i1,image_meanL,image_meanR,P)  


for c in range (2):
    for i in range(1,n1-1):
        for j in range(1,n2-1):
            if divT1[c,i,j]>1.3*(divT1[c,i+1,j]+divT1[c,i-1,j]+divT1[c,i,j+1]+divT1[c,i,j-1])/4 or divT1[c,i,j]<0.7*(divT1[c,i+1,j]+divT1[c,i-1,j]+divT1[c,i,j+1]+divT1[c,i,j-1])/4: 
                divT1[c,i,j]=(divT1[c,i+1,j]+divT1[c,i-1,j]+divT1[c,i,j+1]+divT1[c,i,j-1])/4





for i in range (len(T)):
    plt.hist(np.ravel(divT[i]),bins=100)
    plt.xlim(T[i]*0.9,T[i]*1.1)
    plt.show()
    plt.clf()









meanleft=np.zeros((length,width),float)
meanright=np.zeros((length,width),float)  
division=np.zeros((100,length,width),float)  

# image_mean_BG=np.zeros((x2-x1,y2-y1),float)
# imgBG=SpeFile(repBG+BG)._read()



# l1=250
# l2=279



# img=SpeFile('/media/arun/DATA2/Phosphor thermometry/6september/sansfiltres-60g-2.SPE')._read()
# img= ndimage.rotate(img, 90,axes=(2, 1))

# for i in range(len(img)):
#     img[i]=cv2.GaussianBlur(img[i,:,:],(5,5),0)
    
# img_left=img[:,l1:l1+length,b1:b1+width]
# meanleft=np.mean(img_left,axis=0)

# img_right=img[:,l2:l2+length,b2:b2+width]  
# meanright=np.mean(img_right,axis=0)


# for i in range(10):
#     print(i)
#     division[i]= img_left[i]/img_right[i]
#     # plt.contourf(division[i],levels=np.linspace(0.6,1.9,32))
#     # plt.axis('scaled')
#     # plt.show()
#     # plt.clf()

# meandivision=meanleft/meanright
# plt.contourf(meandivision,levels=np.linspace(0.6,1.9,32))
# plt.axis('scaled')
# plt.show()
# plt.clf()


# std=np.zeros((length,width),float)
# std=np.std(division,axis=0)
# plt.contourf(std,levels=np.linspace(0,0.1,32))
# plt.axis('scaled')
# plt.show()
# plt.clf()


# gildas=std/meandivision
# plt.contourf(gildas,levels=np.linspace(0,0.1,32))
# plt.axis('scaled')
# plt.show()
# plt.clf()


# for i in range(10):
#     eva=division[i]/meandivision
#     # plt.contourf(eva,levels=np.linspace(0.8,1.2,32))
#     # plt.axis('scaled')
#     # plt.show()
#     # plt.clf()


















































# meandiv1=np.zeros(len(T),float)
# imagediv1=np.zeros((len(T),length1,width1),float)
# # meandiv1,imagediv1=compute(image_meanL1,image_meanR1,length1,width1,L,B,T1,i2)  









# img_left=img_left/image_meanL[0]
# img_right=img_right/image_meanR[0]
# divv=img_left/img_right
# plt.contourf(divv,levels = np.linspace(0.5,1.8,12))
# plt.show()



# levels1 = np.linspace(0.5,1.5,40)
# import cv2
# for i in range (len(list_im_mean)):
#     imgx=list_im_mean[i]
#     imgc1=imgx[115:515,362:775]
#     imgc2=imgx[554:954,362:775]
#     div2=imgc1/imgc2
#     print(np.mean(div2))
#     plt.contourf(div2,levels=levels1)
#     plt.colorbar()
#     plt.show()


# meanmoy=[]
# for i in range(8):
#     meanmoy[i]=np.mean(list_im_mean[i][b1:b1+width,l1:l1+length])





# plt.plot(Ta,meandiv,'ro')
# # plt.plot(Tb,mean_div,'bo')
# plt.show()
# P=np.polyfit(Ta,mean_div,1)    


# levels1 = np.linspace(0.5,1.5,40)
# import cv2

#     imgc1=imgx[115:515,362:775]
#     imgc2=imgx[554:954,362:775]
#     div2=imgc1/imgc2
#     print(np.mean(div2))
#     plt.contourf(div2,levels=levels1)
#     plt.colorbar()
#     plt.show()






