#extract masks
import numpy as np
import cv2
import supervision as sv

def process_image_for_masks(segmented_image):

    segmented_image_np = np.array(segmented_image)
    # Assume the last channel is the alpha channel if the image has 4 channels
    if segmented_image_np.shape[2] == 4:
        segmented_image_np = segmented_image_np[:, :, :3]
    # Find unique colors in the image, each unique color corresponds to a unique mask
    unique_colors = np.unique(segmented_image_np.reshape(-1, segmented_image_np.shape[2]), axis=0)
    masks = []
    for color in unique_colors:
        # Create a mask for each unique color
        mask = np.all(segmented_image_np == color, axis=-1)
        masks.append(mask)
    return masks


def filter_masks_by_size(
    masks,
    min_mask_size: tuple[int, int] = (20, 20)):
    size_filtered_masks = []
    # for mask in masks:
        # coords = np.argwhere(mask)  # Get coordinates of all "on" pixels
        # if coords.size > 0:
            # y_min, x_min = coords.min(axis=0)
            # y_max, x_max = coords.max(axis=0)
            # height = y_max - y_min + 1
            # width = x_max - x_min + 1
            # if height >= min_mask_size[0] and width >= min_mask_size[1]:
                # size_filtered_masks.append(mask)
    # return size_filtered_masks
    # Calculate the number of True entries in each array
    counts = [np.sum(mask) for mask in masks]

    # Filter out arrays with counts smaller than 400
    filtered_arrays = [mask for mask, count in zip(masks, counts) if count >= 300]

    # Calculate the number of True entries in filtered arrays
    filtered_counts = [np.sum(array) for array in filtered_arrays]

    # Sort the filtered arrays based on the counts in ascending order
    sorted_filtered_arrays = [x for _, x in sorted(zip(filtered_counts, filtered_arrays), key=lambda pair: pair[0])]
    #astype(bool) olarak cevirmeyi dene
    return sorted_filtered_arrays

