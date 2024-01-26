# Script written by Sietse Dijt for Vlijm group RUG
# 2024-01-07

import numpy as np
import tifffile as tif

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension, SILengthDimension

from functools import partial
from numbers import Number

AXES = 'TZCYX'
COLORMAPS = ['R', 'G', 'B', 'C', 'M', 'Y', 'K', 'W']
COLORMAPS = {x: LinearSegmentedColormap.from_list(name=x, colors=['k', 'w' if x=='K' else x.lower()]) for x in COLORMAPS}

# Can be edited by user
DEFAULT_CMAP = 'gray'
DEFAULT_CMAPS = 'RGBKKKKKKKKKKKKKKKKK'
def set_default_cmap(cmap):
    global DEFAULT_CMAP
    DEFAULT_CMAP = cmap
def set_default_multichannel_cmaps(cmaps):
    global DEFAULT_CMAPS
    DEFAULT_CMAPS = cmaps

MERGE_DTYPE = np.uint16
def set_merge_dtype(dtype):
    global MERGE_DTYPE
    MERGE_DTYPE = dtype

class TimeDimension(_Dimension):
    def __init__(self):
        super().__init__('s')
        for unit, factor in TIME_UNITS.items():
            if unit == 's': continue
            self.add_units(unit, factor)
TIME_UNITS = {'ms': 1e-3, 's': 1, 'min': 60, 'h': 3600}
TIME_DIM = TimeDimension()
LENGTH_DIM = SILengthDimension()

# ImagePlus class
class ImagePlus:
    _data   : np.ndarray      #                           Fixed order: TZCYX (ImageJ default)
    _scales : dict       = {} # dict(str: (float, str)) | Dict axis: (px_size, unit), e.g. {'X': (0.1, 'micron')}
    _labels : list            # list(str)               | List of labels for each channel

    def __init__(self, data:np.ndarray, axes:str, scales:dict=None, labels:list=None):
        '''
        Initializes ImagePlus object.

        Parameters:
            data (np.ndarray): 
                Image data.
            axes (str):
                Order of axes in data (valid axes are T, Z, C, Y, X).
            scales: 
                None: Don't initialize scales.
                dict(str: (float, str)): Dict with axis as key and scale as 
                    value (px_size, unit). Possible to combine axes, 
                    e.g. {'XY': (0.1, 'um')}.
            labels:
                None:       Don't initialize labels.
                list(str):  List of channel labels, should have length of 
                            number of channels.
        '''
        check_arg_axes(axes)
        check_arg_data(data, axes)
        self.set_data(data, axes)

        if scales is not None:
            check_arg_scales(scales)
            self.set_scales(scales)

        check_arg_labels(labels, self)
        self.set_labels(labels)
        
    @staticmethod
    def from_tif(filename):
        '''
        Creates and returns an ImagePlus object from the given tif file.
        The scales and labels that are stored in the file are loaded. 
        
        Note: The first label for each channel image is taken as label for 
        that channel, even if for each frame a different label is given.

            Parameters:
                filename (str): Path to the tif file to read.

            Returns:
                ImagePlus object created from the file.
        '''
        with tif.TiffFile(filename) as file:
            tags = file.pages[0].tags
            ijmetadata = tags.get('IJMetadata')
            if ijmetadata is None:
                return ImagePlus(tif.imread(filename))
            
            description = tags['ImageDescription'].value.split('\n')
            descriptors = dict([x.split('=', 1) for x in description])
            frames, slices, channels = descriptors.get('frames'), descriptors.get('slices'), descriptors.get('channels')
            frames = 1 if frames is None else int(frames)
            slices = 1 if slices is None else int(slices)
            channels = 1 if channels is None else int(channels) 
            
            # Load data
            frame0 = file.pages[0].asarray()
            data = np.zeros((frames, slices, channels, *frame0.shape), dtype=frame0.dtype)
            for t in range(frames):
                for z in range(slices):
                    for c in range(channels):
                        data[t,z,c] = file.pages[(t*slices + z)*channels + c].asarray()

            # Scales
            scales = {}

            # XY-scale
            xres, yres = tags.get('XResolution'), tags.get('YResolution')
            unit = descriptors.get('unit')
            if unit is not None:
                if xres is not None: 
                    scales['X'] = (xres.value[1] / xres.value[0], unit)
                if yres is not None: 
                    scales['Y'] = (yres.value[1] / yres.value[0], unit)

            # Other scales
            info = ijmetadata.value.get('Info')
            if info is not None:
                infos = info.split('\n')
                for x in infos:
                    if not x.startswith('Stack scales: ||'):
                        continue
                    stack_scales = x.split(' || ')[1:]
                    for scale in stack_scales:
                        axis, value = scale.split('=', 1)
                        px_size, unit = value.split(';', 1)
                        scales[axis] = (float(px_size), unit.strip())

            # Labels
            ijlabels = ijmetadata.value.get('Labels')
            if ijlabels is None:
                labels = None
            else:
                labels = ijlabels[:channels]

            return ImagePlus(data, scales=scales, labels=labels)
    @staticmethod
    def from_tifs(filename):
        pass #TODO
    @staticmethod
    def from_msr(filename, config_name):
        # what if config does not exist?
        pass #TODO

    def shape(self, axis=None):
        '''
        Returns shape of data.

            Parameters:
                axis:
                    None: return shape tuple of data as stored
                    str: axis or order of axes to return list in
            
            Returns:
                axis=None: tuple of data as stored
                single axis: size of axis
                multiple axes: list of size of axes in given order
        '''
        if axis is None:
            return self._data.shape
        if len(axis) == 1:
            return self._data.shape[AXES.index(axis)]
        return [self._data.shape[AXES.index(x)] for x in axis]
    
    def data(self, axes=None, reduce=None):
        '''
        Returns data as np.ndarray.

            Parameters:
                axes:
                    None: return data as stored
                    str: order of axes to return data in
                reduce:
                    int: index of slice
                    reduce_func: (np.ndarray, axis) → np.ndarray
                        Function that reduces the dimensions of an ndarray
                        to lose the given axis (int). E.g. np.sum, np.max, ...
                    dict (str: int | reduce_func): index of slice or 
                        reduce_func per axis. Can combine axes, e.g. {'XY': np.sum}
                    
            Returns:
                np.ndarray: image data ordered as axes with hidden axes 
                reduced according to the reduce argument.
        '''

        if axes is None:
            return self._data
        
        # Reorder axes to have requested axes in front
        axes_ = ax2ind(axes)
        hidden_axes_ = ax2ind(other_axes(axes))
        data = self._data.transpose(axes_ + hidden_axes_)

        reduce = self._parse_reduce_arg(reduce)

        # Reduce hidden axes
        for i, axis in zip(np.arange(len(axes_), len(AXES))[::-1], hidden_axes_[::-1]):
            if data.shape[i] == 1:
                data = data.squeeze(axis=i)
                continue

            if isinstance(reduce, dict):
                reduce_method = reduce.get(AXES[axis])
            else:
                reduce_method = reduce

            if isinstance(reduce_method, int):
                data = data.take(indices=reduce_method, axis=i)
            elif callable(reduce_method):
                data = reduce_method(data, axis=i)
            else:
                raise TypeError(f'No known data reduction method provided for axis \'{AXES[axis]}\' (see \'reduce\' arg)')

        return data
    def set_data(self, data:np.ndarray, axes:str=None):
        '''
        Sets the data of the image. Does not update the axes scales.
        Resets labels if the number of channels is different.

        Parameters:
            data (np.ndarray): 
                Image data.
            axes (str):
                Order of axes in data (valid axes are T, Z, C, Y, X).
        '''
        check_arg_axes(axes)
        check_arg_data(data, axes)

        axes_ = ax2ind(axes)
        data = np.expand_dims(data, axis=tuple(np.arange(len(axes_), len(AXES))))
        self._data = data.transpose(np.argsort(axes_ + ax2ind(other_axes(axes))))

        if hasattr(self, '_labels') and len(self._labels) != self.shape('C'):
            self.set_labels(None)

    def scales(self, axis:str=None):
        '''
        Returns scale(s) of axis.

            Parameters:
                axis:
                    None: Return dict of all scales
                    str: axis or order of axes to return list of scales in
            
            Returns:
                axis=None: 
                    Full dict of axis: <scale> 
                single axis:
                    <scale> for given axis 
                multiple axes: 
                    list of <scale> for given axes in order
                Note: <scale> = tuple(float, str) with (px_size, unit) or None if scale for that axis not set.
        '''
        if axis is None:
            return self._scales
        if len(axis) == 1:
            return self._scales.get(axis)
        return [self._scales.get(x) for x in axis]
    def set_scale(self, axis:str, px_size:Number, unit:str):
        '''
        Set scale(s) of axis to (px_size, unit). Does not reset other axes.

        Parameters:
            axis (str):
                Axis (one or multiple) to set the scale for.
            px_size (numeric):
                Size of one pixel.
            unit (str):
                Unit of the pixel size.
        '''
        check_arg_axis(axis, channel_allowed=False)
        if not isinstance(px_size, Number):
            raise TypeError(f"px_size must be numeric not {type(px_size).__name__}")
        if not isinstance(unit, str):
            raise TypeError(f"unit must be str not {type(unit).__name__}")
        
        for x in axis:
            self._scales[x] = (px_size, unit)
    def set_scales(self, scales:dict, append:bool=False):
        '''
        Set the scales according to the given dict.

        Parameters:
            scales (dict(str: (float, str))):
                Dict with axis as key and scale as value (px_size, unit). 
                Can combine axes, e.g. {'XY': (0.1, 'um')}.
            append (bool):
                Whether the dict should be appended to the old scales.
        '''
        check_arg_scales(scales)
        if not isinstance(append, bool):
            raise TypeError(f"append must be bool not {type(append).__name__}")

        if not append:
            self._scales = {}
        for axis, (px_size, unit) in scales.items():
            self.set_scale(axis, px_size, unit)                
    def scale_from_msr(self, filename, config_name):
        # what if config does not exist or does not match?
        pass #TODO

    def labels(self, channels=None):
        '''
        Returns label(s) of the given channel(s)

            Parameters:
                channels:
                    None:       Return list of labels
                    int:        Label of the given channel
                    list(int):  Labels of the given channels in order
            
            Returns:
                list(str) or str depending on [channels] arg. None instead
                of str if no label for that channel.
        '''
        if channels is None:
            return self._labels
        if isinstance(channels, int):
            return self._labels[channels]
        return [self._labels[c] for c in channels]
    def set_labels(self, labels):
        '''
        Set labels of the channels to the given list.

        Parameters:
            labels:
                None:       Reset all labels
                list(str):  List of channel labels, should have length of 
                            number of channels.
        '''
        check_arg_labels(labels, self)

        if labels is None:
            self._labels = [None] * self.shape('C')
        else:
            self._labels = labels
    def labels_from_msr(self, filename, config_name):
        # what if config does not exist or does not match?
        pass #TODO

    # Axis operations (update axis info!) → NO CHANNEL OPERATIONS!
    def sum(self, axis):
        return self.reduce(axis, reduce=np.sum)
    def max(self, axis):
        return self.reduce(axis, reduce=np.max)
    def take(self, axis, slice=0, slice_list_axis=None):
        if isinstance(slice, int):
            return self.reduce(axis, reduce=slice)
        
        if slice_list_axis is None:
            raise ValueError('No slice_list_axis provided for slicing differently along some axis/axes.')
        indices = np.expand_dims(slice, axis=self._axes_to_indices(self._other_axes(axis)))
        data = np.take_along_axis(self._data, indices=indices, axis=AXES.index(axis))
        return ImagePlus(data, AXES, scales=self._scales, labels=self._labels)
    def crop(self, axis, min=None, max=None, size=None, center=None):
        # Take from min (default 0) to max (default final; not included; also works with -1 etc)
        # min/max can be tuples if multiple axes at once [or dict, can combine ]
        if (min is None) + (max is None) + (size is None) + (center is None) > 2:
            raise TypeError('ImagePlus.crop() requires max 2 args of min/max/size/center')
        if center is not None:
            if size is None:
                raise TypeError('ImagePlus.crop() requires size arg if center arg provided')
            center = self._parse_intdictlist_arg(center, axis)
            size = self._parse_intdictlist_arg(size, axis)

            min = {x: int(center[x]-size[x]/2) for x in axis}
            max = {x: int(center[x]+size[x]/2) for x in axis}
        elif size is not None:
            size = self._parse_intdictlist_arg(size, axis)
            if min is not None:
                min = self._parse_intdictlist_arg(min, axis)
                max = {x: int(min[x]+size[x]) for x in axis}
            elif max is not None:
                max = self._parse_intdictlist_arg(max, axis)
                min = {x: int(max[x]-size[x]) for x in axis}
            else:
                raise TypeError('ImagePlus.crop() requires min/max/center arg if size arg provided')
        else:
            min = self._parse_intdictlist_arg(min, axis)
            max = self._parse_intdictlist_arg(max, axis)

        data = self._data
        for x in axis:
            i0, i1 = min.get(x), max.get(x)
            if i0 is None: i0 = 0
            if i1 is None: i1 = self.shape(x)
            if i0 < 0:                          raise ValueError(f'Could not crop along \'{x}\': min < 0')
            if i1 > self.shape(x):              raise ValueError(f'Could not crop along \'{x}\': max > length of axis')
            if i1 % (self.shape(x)+1) <= i0:    raise ValueError(f'ould not crop along \'{x}\': max < min')

            data = data.take(range(i0, i1 % (self.shape(x)+1)), axis=AXES.index(x))

        return ImagePlus(data, AXES, scales=self._scales, labels=self._labels) #TODO
    def bin(self, axis, factor, binning=np.average):
        # Compress axis (or multiple) with factor given by factor (must be divisible by two), if axis none: all axes
        # Update scale!
        return ImagePlus() #TODO
    def reduce(self, axis, reduce):
        # TODO: reduce also accept list (same order as axes!)
        # NOT CHANNEL AXIS
        reduce = self._parse_reduce_arg(reduce)
        data = self._data

        for x in sorted(axis, key=lambda y: AXES.index(y)):
            reduce_method = reduce.get(x) if isinstance(reduce, dict) else reduce
            if not callable(reduce_method):
                raise TypeError(f'No known data reduction method provided for axis \'{x}\' (see \'reduce\' arg)')
            data = reduce_method(data, axis=AXES.index(x))
            data = np.expand_dims(data, AXES.index(x))

        return ImagePlus(data, AXES, scales=self._scales, labels=self._labels)

    def show(self, ax=None, axes=None, reduce=None, channel=None, vmin=None, vmax=None, cmap=None, square_aspect=False, scalebar=False, colorbar=None):
        '''
        Shows image on matplotlib plot.

            Parameters:
                ax: 
                    None: Create fig, ax and return
                    Axes: Show image on given Axes object. No return
                axes:
                    None: Show 'YX' axes by default
                    str:  Two axes to show (first axis vertical, second horizontal) [cannot be 'C']
                reduce:
                    See ImagePlus.data() function
                channel:
                    None: All channels merged (ordered as stored) or single channel if only one channel in image
                    int:  Single channel image of the given channel
                    str:  Single channel image of the channel with given label
                    list(int | str): List of channels for multichannel image with order the same as 
                        the other arguments relating to channels (vmin, vmax, cmap). Channels can be 
                        int (nr of channel) or str (channel label)
                vmin:
                    None: Use channel.min() or vmax if channel.min() larger than vmax
                    int:  Minimum count of colorbar
                    list(int): List of vmin for each channel (ordered as in channel arg). Only for multichannel image
                    dict(str: int): Dict with vmin for each channel label. Only for multichannel image
                vmax: 
                    Same as vmin, but for max count of colorbar
                cmap:
                    None: Single channel → default cmap ('gray'), Multichannel → default cmaps ('RGBKKKK...', 20 channels total)
                          Defaults can be changed through set_default_cmap() and set_default_multichannel_cmaps()
                    str: Single channel → name of mpl colormap or one of 'RGBCMYKW' colormaps (black to respective color, 'K' = 'W')
                         Multichannel → each char is one of 'RGBCMYKW' colormaps
                    Colormap: Only single channel
                        Colormap object, will be resampled with vmin, vmax
                    colors: Only single channel
                        Arg for LinearSegmentedColormap.from_list function. E.g. list of begin and end color of colormap (['k', 'r'] for red colormap).
                        Can use any mpl color (named ones (e.g. 'k', 'C0'), hex codes (e.g. '#FFF', '#A5B6C7'), or RGB tuples (float between 0-1))
                    list(str) or dict(str: str): Only multichannel
                        List of mpl colormap names (or one of 'RGBCMYKW') or dict of channel label: mpl colormap name
                    list(Colormap) or dict(str: Colormap): Only multichannel
                        List or dict (as above) with Colormap objects (will be resampled with vmin,vmax)
                    list(colors) or dict(str: colors): Only multichannel
                        List or dict (as above) with 'colors' arg for LinearSegmentedColormap.from_list function (see above)
                square_aspect (bool):
                    Whether to lock to square pixels. 
                    If False and not equal scales → non-square pixels
                    If True and not equal scales → two scalebars
                scalebar (bool):
                    Whether to plot a scalebar on the image
                colorbar: Only single channel
                    None/False: No colorbar
                    True: Show colorbar (if ax is not None, uses plt.colorbar())
                    matplotlib.figure.Figure: figure to draw colorbar on
                    
            Returns:
                ax=None:    tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
                Otherwise:  No return

        '''

        ax_ = ax
        if ax_ is None:
            fig, ax = plt.subplots()
            ax.axis('off')
        
        if axes is None:
            axes = 'YX'

        if isinstance(channel, (int, str)) or self.shape('C') == 1:
            # single channel
            if channel is None: channel = 0
            if isinstance(channel, str): channel = self._labels.index(channel)
            data = self.data(axes='C'+axes, reduce=reduce)[channel]

            if vmin is None:
                vmin = data.min()
                if not vmax is None and vmin > vmax:
                    vmin = vmax
            if vmax is None:          
                vmax = data.max()
                if vmax < vmin:
                    vmax = vmin

            cmap = self._get_cmap(cmap)
            ax_img = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none')

            if colorbar is not None and colorbar:
                cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.05)

                if ax_ is None:
                    fig.colorbar(ax_img, cax=cax)
                else:
                    if isinstance(colorbar, matplotlib.figure.Figure):
                        colorbar.colorbar(ax_img, cax=cax)
                    else:
                        plt.colorbar(ax_img, cax=cax)
        else:
            # multichannel
            if channel is None: channel = np.arange(self.shape('C'))
            data = self.data(axes='C'+axes, reduce=reduce)

            if cmap is None:
                cmap = DEFAULT_CMAPS

            img = np.zeros((*data.shape[1:], 4))
            for i, ch in enumerate(channel):
                if isinstance(ch, str):
                    ch = self._labels.index(ch)

                if vmin is None:             vmin_ch = None
                elif isinstance(vmin, int):  vmin_ch = vmin
                elif isinstance(vmin, dict): vmin_ch = vmin.get(self._labels[ch])
                else:                        vmin_ch = vmin[ch]

                if vmax is None:             vmax_ch = None
                elif isinstance(vmax, int):  vmax_ch = vmax
                elif isinstance(vmax, dict): vmax_ch = vmax.get(self._labels[ch])
                else:                        vmax_ch = vmax[ch]

                if vmin_ch is None:
                    vmin_ch = data[ch].min()
                    if not vmax_ch is None and vmin_ch > vmax_ch:
                        vmin_ch = vmax_ch
                if vmax_ch is None:          
                    vmax_ch = data[ch].max()
                    if vmax_ch < vmin_ch:
                        vmax_ch = vmin_ch
                
                if isinstance(cmap, str):    cmap_ch = COLORMAPS[cmap[i]]
                elif isinstance(cmap, dict): cmap_ch = cmap[self._labels[ch]]
                else:                        cmap_ch = cmap[i]
                cmap_ch = self._get_cmap(cmap_ch).resampled(np.iinfo(MERGE_DTYPE).max + 1)

                channel_data = np.clip((data[ch].astype(float) - vmin_ch)/(vmax_ch - vmin_ch), 0, 1) * np.iinfo(MERGE_DTYPE).max
                img += cmap_ch(channel_data.astype(MERGE_DTYPE))
            img = np.clip(img, 0, 1)

            ax.imshow(img, interpolation='none')

        # Scalebar and aspect ratio
        scale_ver = self.scales(axes[0])
        scale_hor = self.scales(axes[1])
        if scalebar and (scale_ver is None or scale_hor is None):
            raise ValueError(f'Cannot show scalebar since scale not set for {"axes" if scale_ver is scale_hor else "axis"} \'{axes[0] if scale_ver is None else ""}{axes[1] if scale_hor is None else ""}\'')
        if scale_ver is not None and scale_hor is not None:
            px_ver, unit_ver = scale_ver
            px_hor, unit_hor = scale_hor
            if unit_ver == 'micron': unit_ver = 'um'
            if unit_hor == 'micron': unit_hor = 'um'

            if LENGTH_DIM.is_valid_units(unit_ver) and LENGTH_DIM.is_valid_units(unit_hor):
                aspect = LENGTH_DIM.convert(px_ver, unit_ver, 'm') / LENGTH_DIM.convert(px_hor, unit_hor, 'm')

                if square_aspect and aspect != 1:
                    if scalebar:
                        ax.add_artist(ScaleBar(px_hor, unit_hor, length_fraction=0.2, width_fraction=0.01, color='w', box_alpha=0, location='lower right', scale_loc='left'))
                        ax.add_artist(ScaleBar(px_ver, unit_ver, length_fraction=0.2, width_fraction=0.01, color='w', box_alpha=0, rotation='vertical', location='lower right', scale_loc='top'))
                else:
                    ax.set_aspect(aspect)
                    if scalebar:
                        ax.add_artist(ScaleBar(px_hor, unit_hor, length_fraction=0.2, width_fraction=0.02, color='w', box_alpha=0, location='lower right', scale_loc='top'))
            elif scalebar:
                dimension_hor = TIME_DIM if unit_hor in TIME_UNITS else 'si-length'
                dimension_ver = TIME_DIM if unit_ver in TIME_UNITS else 'si-length'

                ax.add_artist(ScaleBar(px_hor, unit_hor, dimension=dimension_hor, length_fraction=0.2, width_fraction=0.01, color='w', box_alpha=0, location='lower right', scale_loc='left'))
                ax.add_artist(ScaleBar(px_ver, unit_ver, dimension=dimension_ver, length_fraction=0.2, width_fraction=0.01, color='w', box_alpha=0, rotation='vertical', location='lower right', scale_loc='top'))

        if ax_ is None:
            return fig, ax
        
    def save(self, filename, vmin=None, vmax=None, cmap=None, compression=None):
        '''
        Saves the ImagePlus tif data to a tif file.

            Parameters:
                filename (str):
                    Path of the file to save. If it does not end in .tif or .tiff, '.tif' is appended to the name.
                vmin:
                    None: Use channel.min() or vmax if channel.min() larger than vmax
                    int:  Minimum count of colorbar
                    list(int): List of vmin for each channel (ordered as in channel arg). Only for multichannel image
                    dict(str: int): Dict with vmin for each channel label. Only for multichannel image
                vmax: 
                    Same as vmin, but for max count of colorbar
                cmap: Only for multichannel.
                    None: No colormap saved
                    str: Each char is one of 'RGBCMYKW' colormaps (black to respective color, 'K' = 'W')
                    list(str) or dict(str: str): Only multichannel
                        List of mpl colormap names (or one of 'RGBCMYKW') or dict of channel label: mpl colormap name
                    list(Colormap) or dict(str: Colormap): Only multichannel
                        List or dict (as above) with Colormap objects (will be resampled with vmin,vmax)
                    list(colors) or dict(str: colors): Only multichannel
                        List or dict (as above) with 'colors' arg for LinearSegmentedColormap.from_list function,
                        e.g. list of begin and end color of colormap (['k', 'r'] for red colormap).
                        Can use any mpl color (named ones (e.g. 'k', 'C0'), hex codes (e.g. '#FFF', '#A5B6C7'), or RGB tuples (float between 0-1))
                compression:
                    None: No compression
                    str:  Compression type (e.g. 'zlib'). See tifffile.imwrite for more info.
        '''
        if not (filename.endswith('.tif') or filename.endswith('.tiff')):
            filename += '.tif'

        # cmap only for multichannel!
        frames, slices, channels = self.shape('TZC')
        images = frames * slices * channels

        # Scales
        scale_x, scale_y = self.scales('XY')
        if scale_x is None or scale_y is None:
            px_x, px_y, unit = 1, 1, ' '
        else:
            px_x, unit_x = scale_x
            px_y, unit_y = scale_y
            px_x = LENGTH_DIM.convert(px_x, 'um' if unit_x=='micron' else unit_x, 'um')
            px_y = LENGTH_DIM.convert(px_y, 'um' if unit_y=='micron' else unit_y, 'um')
            unit = 'micron'
        scale_t, scale_z = self.scales('T'), self.scales('Z')
        info = '\nStack scales:'
        if scale_t is not None:
            info += f' || T={scale_t[0]};{scale_t[1]}'
        if scale_z is not None:
            info += f' || Z={scale_z[0]};{scale_z[1]}'
        info += '\n'

        # Image description and Info
        channels_str = f'channels={channels}\n'
        slices_str   = f'slices={slices}\n'
        frames_str   = f'frames={frames}\n'
        images_str   = f'images={frames*slices*channels}\n{channels_str if channels>1 else ""}{slices_str if slices>1 else ""}{frames_str if frames>1 else ""}'
        hyperstack_str = 'hyperstack=true\n'

        tags = {}
        if len(info) > 15:
            tags['Info'] = info

        if channels == 1:
            # Single channel
            if vmin is None: vmin = self._data.min()
            if vmax is None: vmax = self._data.max()

            if self.labels(0) is not None:
                tags['Labels'] = [self.labels(0) for _ in range(images)]

            tif.imwrite(filename, self._data.squeeze(), metadata=None, resolution=(1/px_x, 1/px_y), compression=compression,
                        description=f'ImageJ=1.54g\n{images_str}{hyperstack_str if slices>1 and frames>1 else ""}\nunit={unit}\ncf=0\nc0=-32768\nc1=1\nvunit=gray value\nmin={vmin+32768}\nmax={vmax+32768}',
                        extratags=tif.imagej_metadata_tag(tags, '<')
            )
        else:
            # Multichannel
            vmins, vmaxs = [], []
            for ch in range(channels):
                if vmin is None:             vmin_ch = None
                elif isinstance(vmin, int):  vmin_ch = vmin
                elif isinstance(vmin, dict): vmin_ch = vmin.get(self._labels[ch])
                else:                        vmin_ch = vmin[ch]
                if vmin_ch is None:          vmin_ch = self._data.take(ch, AXES.index('C')).min()
                vmins.append(vmin_ch)

                if vmax is None:             vmax_ch = None
                elif isinstance(vmax, int):  vmax_ch = vmax
                elif isinstance(vmax, dict): vmax_ch = vmax.get(self._labels[ch])
                else:                        vmax_ch = vmax[ch]
                if vmax_ch is None:          vmax_ch = self._data.take(ch, AXES.index('C')).max()
                vmaxs.append(vmax_ch)
            tags['Ranges'] = [[val+32768 for pair in zip(vmins, vmaxs) for val in pair]]
            if None not in self.labels():
                tags['Labels'] = [self.labels(i) for _ in range(frames*slices) for i in range(channels)]
            
            if cmap is not None:
                cmaps = []
                for ch in range(channels):
                    if isinstance(cmap, str):    cmap_ch = COLORMAPS[cmap[ch]]
                    elif isinstance(cmap, dict): cmap_ch = cmap[self._labels[ch]]
                    else:                        cmap_ch = cmap[ch]
                    cmaps.append(self._get_cmap(cmap_ch))
                tags['LUTs'] = [(x.resampled(256)(np.arange(256))[:,:3] * 255).astype(np.uint8).transpose() for x in cmaps]

            tif.imwrite(filename, self._data.squeeze(), metadata=None, resolution=(1/px_x, 1/px_y), compression=compression,
                        description=f'ImageJ=1.54g\n{images_str}{hyperstack_str if slices>1 or frames>1 else ""}mode=composite\nunit={unit}\ncf=0\nc0=-32768\nc1=1\nvunit=gray value\nmin={vmins[0]+32768}\nmax={vmaxs[0]+32768}',
                        extratags=tif.imagej_metadata_tag(tags, '<')
            )

    # --- Private functions --- TODO: Move to util functions! 
    def _get_cmap(self, cmap):
        if cmap is None: 
            return DEFAULT_CMAP
        if isinstance(cmap, str):
            if cmap in COLORMAPS:
                return COLORMAPS[cmap]
            return plt.get_cmap(cmap)
        if isinstance(cmap, Colormap):
            return cmap
        return LinearSegmentedColormap.from_list('custom', colors=cmap)
    
    def _parse_reduce_arg(self, reduce):
        if not isinstance(reduce, dict):
            return self._parse_reduce_method(reduce)
        
        reduce_parsed = {}
        for axes, reduce_method in reduce.items():
            for x in axes:
                reduce_parsed[x] = self._parse_reduce_method(reduce_method)
        return reduce_parsed
    def _parse_reduce_method(self, reduce_method):
        if isinstance(reduce_method, int):
            return partial(np.take, indices=reduce_method)
        return reduce_method
    
    def _parse_intdictlist_arg(self, arg, axis):
        if arg is None:
            return {}
        if isinstance(arg, int):
            arg = {axis: arg}
        if not isinstance(arg, dict):
            arg = {x: arg[i] for i,x in enumerate(axis)}
        parsed = {}
        for axes, v in arg.items():
            for x in axes:
                parsed[x] = v
        return parsed


# --- Util functions ---
def ax2ind(axes):
    return [AXES.index(x) for x in axes]
def ind2ax(indices):
    return [AXES[x] for x in indices]
def other_axes(axes):
    return ''.join(sorted(set(AXES)-set(axes)))
    

# --- Arg check functions ---
def is_list(object):
    return isinstance(object, (list, tuple, np.ndarray))

def check_arg_axes(axes):
    if not isinstance(axes, str):
        raise TypeError(f"axes must be str, not {type(axes).__name__}")
    if sum([x not in AXES for x in axes]) == 1:
        raise ValueError(f"'{[x for x in axes if x not in AXES][0]}' not a valid axis. Valid axes are T, Z, C, Y, X")
    if sum([x not in AXES for x in axes]) > 1:
        raise ValueError(f"'{', '.join([x for x in axes if x not in AXES])}' not valed axes. Valid axes are T, Z, C, Y, X")
    if len(axes) > len(set(axes)):
        raise ValueError(f"duplicate axis in argument ({axes})")

def check_arg_data(data, axes):
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be numpy.ndarray, not {type(data).__name__}")
    if data.ndim > 5:
        raise ValueError(f"data must have 5 or less dims, not {data.ndim}")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"data dtype must be numeric, not {data.dtype}")
    if data.ndim != len(axes):
        raise ValueError(f"number of specified axes ({len(axes)}) does not match with data dimensions ({data.ndim})")

def check_arg_axis(axis, channel_allowed=True):
    if not isinstance(axis, str):
        raise TypeError(f"axis must be str, not {type(axis).__name__}")
    allowed_axes = (AXES if channel_allowed else other_axes('C'))
    if sum([x not in allowed_axes for x in axis]) == 1:
        raise ValueError(f"'{[x for x in axis if x not in allowed_axes][0]}' not a valid axis {'' if channel_allowed else 'here'}. Valid axes are T, Z, {'C, ' if channel_allowed else ''}Y, X")
    if sum([x not in allowed_axes for x in axis]) > 1:
        raise ValueError(f"'{', '.join([x for x in axis if x not in allowed_axes])}' not valed axes {'' if channel_allowed else 'here'}. Valid axes are T, Z, {'C, ' if channel_allowed else ''}Y, X")
    if len(axis) > len(set(axis)):
        raise ValueError(f"duplicate axis in argument ({axis})")

def check_arg_scales(scales):
    if not isinstance(scales, dict):
        raise TypeError(f"scales must be dict not {type(scales).__name__}")
    for key, value in scales.items(): 
        check_arg_axis(key, channel_allowed=False)
        if not isinstance(value, tuple):
            raise TypeError(f"scales dict values must be tuples of 2 values (px_size, unit), not {type(value).__name__}")
        if len(value) != 2:
            raise TypeError(f"scales dict values must be tuples of 2 values (px_size, unit), not of {len(value)} values")
    
def check_arg_labels(labels, imp):
    if labels is None:
        return
    if not is_list(labels):
        raise TypeError(f"labels must be None or list, not {type(labels).__name__}")
    for label in labels:
        if not isinstance(label, str):
            raise TypeError(f"label must be str, not {type(label).__name__}")
    if len(labels) != imp.shape('C'):
        raise ValueError(f"length of labels ({len(labels)}) does not match number of channels ({imp.shape('C')})")
    if len(labels) == len(set(labels)) + 1:
        raise ValueError(f"duplicate label ('{[x for x in labels if labels.count(x) > 1][0]}')")
    if len(labels) > len(set(labels)) + 1:
        raise ValueError(f"duplicate labels ({set((x for x in labels if labels.count(x) > 1))})")
        