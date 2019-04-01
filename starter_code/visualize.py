from pathlib import Path
import argparse

import scipy.misc
import numpy as np

from starter_code.utils import load_case


# Constants
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_TUMOR_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3


def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)   #(a, a_min, a_max, out=None) if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        											#SCALE between 0 and 1
    # Scale to values between 0 and 1
    mxval = np.max(volume)   
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume	
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros
    shp = segmentation.shape
    print("shp dim", shp)
    print(shp[0], shp[1], shp[2], 3)
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location
    seg_color[np.equal(segmentation,1)] = k_color   #for kidney
    seg_color[np.equal(segmentation,2)] = t_color   #for Tumor
    print("seg_color" , seg_color.shape)
    return seg_color


def overlay(volume_ims, segmentation_ims, segmentation, alpha):
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha*segmentation_ims+(1-alpha)*volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )
    return overlayed


def visualize(cid, destination, hu_min=DEFAULT_HU_MIN, hu_max=DEFAULT_HU_MAX, 
    k_color=DEFAULT_KIDNEY_COLOR, t_color=DEFAULT_TUMOR_COLOR,
    alpha=DEFAULT_OVERLAY_ALPHA):
    # Prepare output location
    out_path = Path(destination)
    if not out_path.exists():
        out_path.mkdir()  

    # Load segmentation and volume
    vol, seg = load_case(cid)
    vol = vol.get_data()
    seg = seg.get_data()
    seg = seg.astype(np.int32)
    
    # Convert to a visual format
    vol_ims = hu_to_grayscale(vol, hu_min, hu_max)   #gray scale image from original image.
    seg_ims = class_to_color(seg, k_color, t_color)  #a new image with appropriate color for kidney and tumor
    
    # Overlay the segmentation colors
    viz_ims = overlay(vol_ims, seg_ims, seg, alpha)  # alpha = 0.3, overlay images using some formula

    # Save individual images to disk
    for i in range(viz_ims.shape[0]):      			 #save images in png format
        fpath = out_path / ("{:05d}.png".format(i))	 
        scipy.misc.imsave(str(fpath), viz_ims[i])


if __name__ == '__main__':
    # Parse command line arguments
    desc = "Overlay a case's segmentation and store it as a series of pngs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-c", "--case_id", required=True,
        help="The identifier for the case you would like to visualize"
    )
    parser.add_argument(
        "-d", "--destination", required=True,
        help="The location where you'd like to store the series of pngs"
    )
    parser.add_argument(
        "-u", "--upper_hu_bound", required=False, default=DEFAULT_HU_MAX,
        help="The upper bound at which to clip HU values"
    )
    parser.add_argument(
        "-l", "--lower_hu_bound", required=False, default=DEFAULT_HU_MIN,
        help="The lower bound at which to clip HU values"
    )
    args = parser.parse_args()

    # Run visualization
    visualize(
        args.case_id, args.destination, 
        hu_min=args.lower_hu_bound, hu_max=args.upper_hu_bound
    )
    
    
    
    
