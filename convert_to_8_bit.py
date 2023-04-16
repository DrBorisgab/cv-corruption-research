import rasterio
from aeronet.dataset import Band
from aeronet.converters.split import split
import numpy as np

# Normalization parameter; change to 2 to make output more contrast
WIDTH = 3


def main(ms_file, out_file, r=1, g=2, b=3):
    print("Running image preprocessing script")
    with rasterio.open(ms_file) as src:
        profile = src.profile
        red = src.read(r)
        grn = src.read(g)
        blu = src.read(b)

    profile.update(count=3, dtype='uint8')
    
    # Nodata should be transfered to uint8 range to match the image dtype
    nodata_value = profile.get('nodata')
    if nodata_value is not None:
        if 0 <= nodata_value <= 255:
            profile.update(nodata=int(nodata_value))
        else:
            profile.update(nodata=0)

    channels_8bit = []
    for channel in [red, grn, blu]:
        mean, std, min_val, max_val = np.mean(channel), np.std(channel), np.min(channel), np.max(channel)
        m = max(min_val, mean - WIDTH*std)
        M = min(max_val, mean + WIDTH*std)
        ch_8bit = np.floor_divide(
            np.multiply((channel - m), 255, dtype='float32'),
            (M-m)
        )
        # We clip it from 1 to leave 0 value for nodata
        ch_8bit = np.clip(np.around(ch_8bit, 0), 1, 255).astype('uint8')
        channels_8bit.append(ch_8bit)

    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(channels_8bit[0], 1)
        dst.write(channels_8bit[1], 2)
        dst.write(channels_8bit[2], 3)
    print('Done')