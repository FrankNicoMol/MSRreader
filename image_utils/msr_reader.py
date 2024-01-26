# Script written by Sietse Dijt for Vlijm group RUG
# 2024-01-07

# MSR format description: 
#   https://imspectordocs.readthedocs.io/en/latest/fileformat.html

from .image_settings import ImageSettings, Gating
from .image_plus import ImagePlus, AXES

import numpy as np
import zlib

from struct import unpack
import xml.etree.ElementTree as ET

OMAS_BF_MAX_DIMENSIONS = 15
READER_FORMAT_VERSION = 6
READER_MIN_VERSION = 4 # below version 4 no tag dictionary

class ConfigNotFoundError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(message)

def _uint(f):
    return unpack('<I', f.read(4))[0]
def _ulong(f):
    return unpack('<Q', f.read(8))[0]
def _double(f):
    return unpack('<d', f.read(8))[0]
def _string(f, length):
    return f.read(length)
def _string_decode(f, length):
    return f.read(length).decode()
def _si_fraction(f):
    return unpack('<ii', f.read(8))[0]
def _si_value(f):
    exponents = [_si_fraction(f) for _ in range(9)]
    scale_factor = _double(f)
    return exponents, scale_factor
def _len_string_decode(f):
    return _string_decode(f, _uint(f))

def _read_header(f):
    magic_header = _string(f, 10)
    assert magic_header == b'OMAS_BF\n\xff\xff'
    
    format_version = _uint(f)
    first_stack_pos = _ulong(f)
    descr_len = _uint(f)
    description = _string_decode(f, descr_len)

    if format_version >= 2:
        meta_data_position = _uint(f)

    return first_stack_pos
def _read_stack(f):
    # --- Reading start ---
    magic_header = _string(f, 16)
    assert magic_header == b'OMAS_BF_STACK\n\xff\xff'

    format_version = _uint(f)
    assert format_version >= READER_MIN_VERSION
    rank = _uint(f)
    res = [_uint(f) for _ in range(OMAS_BF_MAX_DIMENSIONS)][:rank]
    length = [_double(f) for _ in range(OMAS_BF_MAX_DIMENSIONS)][:rank]
    offset = [_double(f) for _ in range(OMAS_BF_MAX_DIMENSIONS)][:rank]
    dt = _uint(f)
    compression_type = _uint(f)
    compression_level = _uint(f)
    name_len = _uint(f)
    descr_len = _uint(f)
    reserved = _ulong(f)
    data_len_disk = _ulong(f)
    next_stack_pos = _ulong(f)

    name = _string_decode(f, name_len)
    description = _string_decode(f, descr_len)

    if name.endswith('[Pop]'): # line profile → skip
        return None, next_stack_pos

    # Data
    start_pos = f.tell()
    data = f.read(data_len_disk)
    
    # Stack footer
    start_fp = f.tell()

    if format_version >= 1: 
        size = _uint(f)
        has_col_positions = [_uint(f) for _ in range(OMAS_BF_MAX_DIMENSIONS)]
        has_col_labels = [_uint(f) for _ in range(OMAS_BF_MAX_DIMENSIONS)]
        metadata_length = _uint(f)
    if format_version >= 2:
        si_value = _si_value(f)
        si_dimensions = [_si_value(f) for _ in range(OMAS_BF_MAX_DIMENSIONS)]
    if format_version >= 3:
        num_flush_points = _ulong(f)
        flush_block_size = _ulong(f)
    if format_version >= 4:
        tag_dict_length = _ulong(f)
    if format_version >= 5:
        stack_end_disk = _ulong(f)
        min_format_version = _uint(f)
        stack_end_used_disk = _ulong(f)
        assert READER_FORMAT_VERSION >= min_format_version
    if format_version >= 6:
        samples_written = _ulong(f)
        num_chunk_positions = _ulong(f)

    # Variable-sized stack footer
    if format_version >= 1:
        f.seek(start_fp + size) # go to variable-sized part of footer

        label_strings = [_len_string_decode(f) for _ in range(rank)]
        col_positions = [(_double(f) if has_col_positions[i] != 0 else None)          for i in range(OMAS_BF_MAX_DIMENSIONS)]
        col_labels    = [(_len_string_decode(f)  if has_col_labels[i] != 0 else None) for i in range(OMAS_BF_MAX_DIMENSIONS)]
        metadata      = _string_decode(f, metadata_length)
    if format_version >= 3:
        flush_points  = [_ulong(f) for _ in range(num_flush_points)]
    if format_version >= 4:
        start_fp = f.tell()
        tag_dict = {}
        while f.tell() - start_fp < tag_dict_length - 4:
            key = _len_string_decode(f)
            val = _len_string_decode(f)
            tag_dict[key] = val
    if format_version >= 6:
        chunk_positions = [(_ulong(f), _ulong(f)) for _ in range(num_chunk_positions)]

        pos, idx = 0, 0
        f.seek(start_pos)
        while pos < samples_written:
            bytes_to_read = samples_written - pos
            seek_pos = -1
            if idx < num_chunk_positions:
                if pos + bytes_to_read > chunk_positions[idx][0]:
                    bytes_to_read = chunk_positions[idx][0] - pos
                    seek_pos = chunk_positions[idx][1] + start_pos
                    idx += 1
            if bytes_to_read > 0:
                data += f.read(bytes_to_read)
            if seek_pos != -1:
                f.seek(seek_pos)
            pos += bytes_to_read
    # --- Reading done ---
    
    # Parse data (pad with zeros if incomplete, i.e. measurement not finished)
    if compression_type == 1:
        data = zlib.decompress(data)
    data = np.frombuffer(data, '<h')

    finished = data.shape[0] == np.prod(res[:rank])
    if not finished:
        data = np.pad(data, (0, np.prod(res[:rank])-data.shape[0]), 'constant')
    data = data.reshape(res[:rank], order='F')

    return {
        'name': name,
        'data': data,
        'labels': label_strings,
        'finished': finished,
        'length': length,
        'offset': offset,
        'tag_dict': tag_dict,
    }, next_stack_pos
def _read_msr(filename):
    stacks = []
    with open(filename, 'rb') as f:
        stack_pos = _read_header(f)
        while stack_pos != 0:
            f.seek(stack_pos)
            stack, stack_pos = _read_stack(f)
            if stack != None:
                stacks.append(stack)
    return stacks

def _parse_stack_data(stack, silent=False):
    data = stack['data']    
    axes = [label[-1] for label,res in zip(stack['labels'], data.shape) if res > 1]
    lengths = [l/res for l,res in zip(stack['length'], data.shape) if res > 1]
    data = np.squeeze(data)

    if not silent and not stack['finished']:
        print(f"Warning: Unfinished measurement ({stack['name']})")
    
    return data, ''.join(axes), {x: (l, 's' if x == 'T' else 'm') for x,l in zip(axes, lengths)}
def _get_config_name(stack):
    root = ET.fromstring(stack['tag_dict']['imspector'])
    return root.find('old_style/Measurement/doc/propset_label').text
def _get_channel_names(stack):
    root = ET.fromstring(stack['tag_dict']['imspector'])
    channels = root.findall('doc/ExpControl/measurement/channels/item')
    return [channel.find('name').text for channel in channels]
def _parse_stack_settings(stack):
    root = ET.fromstring(stack['tag_dict']['imspector'])

    settings = ImageSettings()
    settings.microscope = 'Abberior'
    settings.objective = root.find('doc/IX83/light_path/objlens').text
    settings.config_name = root.find('old_style/Measurement/doc/propset_label').text

    # -- Imspector 'Scan Range' window
    _, _, settings.pixel_sizes = _parse_stack_data(stack, silent=True)

    settings.bidirectional_scan = root.find('doc/Measurement/axes/bidirectional_scan').text == '1'
    settings.fast_scan = root.find('doc/ExpControl/scan/range/fast_scan/active').text == '1'

    # Stage positions (coarse + fine offsets)
    settings.stage_x = float(root.find('doc/ExpControl/scan/range/coarse_x/g_off').text) + float(root.find('doc/ExpControl/scan/range/x/g_off').text)
    settings.stage_y = float(root.find('doc/ExpControl/scan/range/coarse_y/g_off').text) + float(root.find('doc/ExpControl/scan/range/y/g_off').text)
    settings.stage_z = float(root.find('doc/ExpControl/scan/range/coarse_z/g_off').text) + float(root.find('doc/ExpControl/scan/range/z/g_off').text)
    for label, offset in zip(stack['labels'], stack['offset']):
        if label[-1] == 'X': settings.offset_x = offset
        if label[-1] == 'Y': settings.offset_y = offset
        if label[-1] == 'Z': settings.offset_z = offset

    settings.autofocus_on = root.find('hwr/IX83/af/keep_af_focus_pos').text == '1'
    settings.rotation, settings.roll, settings.tilt = tuple(map(float, root.find('doc/ExpControl/scan/range/orientation').text.split()))

    settings.dwelltime = float(root.find('doc/ExpControl/measurement/channels/item/dwelltime').text)
    # TODO: frame trigger?, extend until stopped?, stepwise scanning?

    # -- Imspector 'Line Steps' window
    settings.line_repetitions = int(root.find('doc/ExpControl/measurement/line_steps/repetitions').text)
    settings.line_interleave = root.find('doc/ExpControl/measurement/line_steps/interleave').text == '1'

    linesteps_active = [x == '1' for x in root.find('doc/ExpControl/measurement/line_steps/step_active').text.split()]
    settings.linesteps = [int(x) for x,active in zip(root.find('doc/ExpControl/measurement/line_steps/step_duration').text.split(), linesteps_active) if active]

    pixelsteps_active = [x == '1' for x in root.find('doc/ExpControl/measurement/pixel_steps/step_active').text.split()]
    settings.pixelsteps = [float(x) for x,active in zip(root.find('doc/ExpControl/measurement/pixel_steps/step_duration').text.split(), pixelsteps_active) if active]

    # -- Imspector 'Channels' window
    channels = root.findall('doc/ExpControl/measurement/channels/item')
    for channel in channels:
        name = channel.find('name').text

        detector = channel.find('detsel/detector').text
        spectral_min = round(1e9*float(channel.find('detsel/spectral_range/min').text))
        spectral_max = round(1e9*float(channel.find('detsel/spectral_range/max').text))

        lasers_names = [x.text for x in root.findall('hwr/ExpControl/devices/lasers/lasers/item/name')]
        lasers_power = [float(x.text) for x in channel.findall('lasers/item/power/calibrated')]
        lasers_active = [x.text == '1' for x in channel.findall('lasers/item/active')]
        lasers = {name: power for name,power,active in zip(lasers_names, lasers_power, lasers_active) if active}
        
        gating = Gating(
            active = channel.find('gating/active').text == '1',
            delay  = float(channel.find('gating/delay').text),
            width  = float(channel.find('gating/width').text)
        )

        channel_linesteps = [i for i,x in enumerate(channel.find('steps/linestep_active').text.split()) if x=='1' and linesteps_active[i]]
        channel_pixelsteps = [i for i,x in enumerate(channel.find('steps/pixelstep_active').text.split()) if x=='1' and pixelsteps_active[i]]
        settings.add_channel(name, detector, spectral_min, spectral_max, lasers, gating, channel_linesteps, channel_pixelsteps)
        # TODO: Pulse Steps?, Mask Stack, Use Probe Channel, Rescue

    # -- Imspector 'Pinhole' window
    settings.pinhole = float(root.find('doc/Pinhole/size_used_au').text)

    # -- Imspector 'Measurement Setup' window
    settings.scan_axes = root.find('doc/ExpControl/scan/range/scanmode').text.upper()

    # -- Imspector 'Measurement' window
    settings.split3D = float(root.findall('doc/ExpControl/measurement/lasers/item/three_d/power_dist')[-1].text)
    return settings

def _construct_imp(stack_list, channels=None):
    datas = []
    for stack in stack_list:
        data, axes, scales = _parse_stack_data(stack)
        datas.append(data)
    data = np.stack(datas)

    if channels is None:
        channels = _get_channel_names(stack_list[0])

    return ImagePlus(data, 'C'+axes, scales, labels=channels)

def read_msr_imps(filename:str, config:str=None):
    '''
    Reads, parses and returns the images in the msr file as ImagePlus objects.

    Parameters:
        filename (str):
            Path to the msr file.
        config:
            None: Return dict for all configs
            str: Return single specified config
        
    Returns:
        config=None: dict(str: ImagePlus)
            Dict of {config_name: ImagePlus} for each config.
        specified config: ImagePlus
            ImagePlus object for the given config. If the config is not found,
            throws a ConfigNotFoundError (custom).
    '''
    stacks = _read_msr(filename)

    configs = {}
    for stack in stacks:
        conf = _get_config_name(stack)
        if conf not in configs:
            configs[conf] = []
        configs[conf].append(stack)
    
    if config is None:
        return {conf: _construct_imp(stack_list) for conf, stack_list in configs.items()}
    
    if config not in configs:
        raise ConfigNotFoundError(f"File ({filename}) did not contain config '{config}'")
    return _construct_imp(configs[config])
def read_msr_configs(filename:str, config:str=None):
    '''
    Reads, parses and returns an ImageSettings (containing image acquisition 
    parameters) and an ImagePlus (containing image data) object for each image 
    in the msr file.

    Parameters:
        filename (str):
            Path to the msr file.
        config:
            None: Return dict for all configs
            str: Return single specified config
        
    Returns:
        config=None: dict(str: (ImageSettings, ImagePlus))
            Dict of {config_name: (ImageSettings, ImagePlus)} for each config.
        specified config: (ImageSettings, ImagePlus)
            Tuple of ImageSettings and ImagePlus object. If the config is not
            found, throws a ConfigNotFoundError (custom).
    '''
    stacks = _read_msr(filename)

    configs = {}
    for stack in stacks:
        conf = _get_config_name(stack)
        if conf not in configs:
            configs[conf] = []
        configs[conf].append(stack)
    
    if config is None:
        result = {}
        for config, stack_list in configs.items():
            settings = _parse_stack_settings(stack_list[0])
            imp = _construct_imp(stack_list, channels=settings.channel_names())
            result[config] = (settings, imp)
        return result
    
    if config not in configs:
        raise ConfigNotFoundError(f"File ({filename}) did not contain config '{config}'")
    settings = _parse_stack_settings(configs[config][0])
    imp = _construct_imp(configs[config], channels=settings.channel_names())
    return (settings, imp)

def show_msr(filename):
    # show all images and basic settings in the msr file to easily select
    # which config you need 
    pass #TODO