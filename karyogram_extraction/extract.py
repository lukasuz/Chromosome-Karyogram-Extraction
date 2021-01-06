import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import floor
import argparse

structure = np.array([5, 7, 6, 5]) * 2 # amount of chromosomes for each line
chromosome_tags = np.array(range(1,24)) # basically the name that the ordered chromosomes are going to get, i.e. 1-23

opening_kernel = np.ones((3,3), np.uint8)
# erosion_kernel = np.ones((7,1), np.uint8)
# dilation_kernel = erosion_kernel

def get_labels(path):
    """ Read a chromosome image and returns its connected component labels matrix

    Arguments:
        path: path to image

    Returns:
        number of labels, labels
    """
    img = cv.imread(path, 0)
    # Thresholding seems
    avg = np.average(img) 
    img = cv.threshold(img, avg - 1, 255, cv.THRESH_BINARY)[1]
    # Flip values
    img = 255 - img
    # Morph. Opening for removing noise and detaching chromosomes
    img = cv.morphologyEx(img, cv.MORPH_OPEN, opening_kernel)
    # img = cv.morphologyEx(img, cv.MORPH_ERODE, erosion_kernel)
    num_labels, labels_im  = cv.connectedComponents(img, connectivity=8)

    return num_labels, labels_im

def make_one_hot_labels(num_labels, labels):
    """ Creates one hot representation of the labels, where each channel
        represents the label mat of a single label value

    Arguments:
        num_labels: number of connected component labels
        labels: connected component labels
    """
    one_hot_labels = np.zeros((labels.shape[0], labels.shape[1], num_labels))
    for i in range(num_labels):
        one_hot = labels == i
        one_hot_labels[:,:,i] = one_hot

    return one_hot_labels

def filter_labels(oh_labels, min_area):
    """ Filters the labels:
        - Thresholds the labels given a minimum volume value
        - removes background and the line at the bottom
    
    Arguments:
        oh_labels: matrix of one hot connected component labels
        min_area: minimum pixel are of a component

    Returns:
        Filtered one hot encoded connected component labels

    """
    pixels = oh_labels.shape[0] * oh_labels.shape[1]
    thres = pixels * min_area
    indices = np.indices((oh_labels.shape[0], oh_labels.shape[1]))

    filtered_oh_labels = np.zeros_like(oh_labels)
    counter = 0
    for i in range(oh_labels.shape[2]):
        # true when the densitiy of the blob is bigger than the threshold
        density_bigger_than_thres = np.sum(oh_labels[:,:,i]) > thres
        # true if the blob touches the right border of the img
        # which is the case for the background and the line
        touches_border = np.max(indices[1] * oh_labels[:,:,i]) + 1 == oh_labels.shape[1]
        if density_bigger_than_thres and not touches_border:
            filtered_oh_labels[:,:,counter] = oh_labels[:,:,i]
            counter += 1

    # reduce size accourding to remaining labels
    filtered_oh_labels = filtered_oh_labels[:,:,0:counter]
    return filtered_oh_labels

def extract_label_centroids(oh_labels):
    """ Calculates the centroid of each one hotted label

    Arguments:
        oh_labels: matrix of one hot connected component labels

    Returns:
        A 2D array with component centroids
    """
    indices = np.indices((oh_labels.shape[0], oh_labels.shape[1]))
    centroids = np.zeros((2, oh_labels.shape[2]))
    for i in range(oh_labels.shape[2]):
        y_mean = int(np.round(np.sum(oh_labels[:,:,i] * indices[0]) / np.sum(oh_labels[:,:,i])))
        x_mean = int(np.round(np.sum(oh_labels[:,:,i] * indices[1]) / np.sum(oh_labels[:,:,i])))
        centroids[0,i] = x_mean
        centroids[1,i] = y_mean

    return centroids

def order_components(oh_labels, centroids, structure=structure):
    """ Orders the components according to their chromosome number.

    Arguments:
        oh_labels: matrix of one hot connected component labels.
        centroids: centroids of the connected components.
        structure: arrangement of chromosomes on the image

    Returns:
        The ordered oh labels
    """
    
    ordered = np.zeros_like(oh_labels)
    x = np.copy(centroids[0,:])
    y = np.copy(centroids[1,:])

    y_argsort = np.argsort(y)

    for i in range(len(structure)):
        amount = int(structure[i])
        prev_amount = int(np.sum(structure[0:i]))

        # get all indx elements in "one line" on the y axis 
        subset_indx = []
        for j in range(prev_amount, prev_amount + amount):
            indx = np.where(y_argsort == j)
            indx = indx[0][0] # weird access
            subset_indx.append(indx)

        # get all the corresponding x values from the indx subset
        subset_x = [x[indx] for indx in subset_indx]
        subset_x_argsort = np.argsort(subset_x) 

        # order the the labels according to x and y position
        for k in range(len(subset_x_argsort)):
            ordered[:,:,prev_amount+k] = oh_labels[:,:,subset_indx[subset_x_argsort[k]]]
    
    return ordered

def pair_components(oh_labels):
    """ Pairs two chromosome labels. Assumes sorted list and only two chromosome per num

    Arguments:
        oh_labels: matrix of one hot connected component labels

    Returns:
        paried oh_labels
    """
    paired = np.zeros((oh_labels.shape[0], oh_labels.shape[1], int(oh_labels.shape[2]/2)))
    for i in range(0, oh_labels.shape[2], 2):
        paired[:,:,int(i/2)] = oh_labels[:,:,i] + oh_labels[:,:,i+1]
    return paired

def extract_bounding_boxes(oh_labels):
    """ Extracts the bounding box parameters for the chromosome labels

    Arguments:
        oh_labels: matrix of one hot connected component labels

    Returns:
        Numpy matrix of bounding box coordinates for each oh label
    """
    boxes = np.zeros((4, oh_labels.shape[2]), dtype="int")
    for i in range(0, oh_labels.shape[2]):
        boxes[:,i] = extract_bounding_box(oh_labels[:,:,i])
    return boxes

def extract_bounding_box(img):
    """ Extract the bounding box params for a binary mask.
        Found on: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    Arguments:
        img: the image

    Returns:
        Bounding box for a single image
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return np.array([rmin, rmax, cmin, cmax], dtype="int")

def extract_and_save_chromsomes(path, bounding_boxes, oh_labels, paired, save_path, tags=chromosome_tags):
    """ Extracts chromosomes given an image path and bounding boxes

    Arguments:
        path: path to image
        bounding_boxes: bounding boxes of each oh label.
        oh_labels: matrix of one hot connected component labels.
        paired: bool, whether chromosome pairs should be extracted in pairs
        save_path: saving path
        tags: chromosome class tags in a list
    """
    img = cv.imread(path)
    temp = path.split("/")[-1]
    fname, ftype = os.path.splitext(temp)
    ftype = 'png' # just save everything as png, not original format

    max_x, max_y = get_max_bounding_box_dims(bounding_boxes)

    if paired:
        for i in range(len(tags)):
            rmin, rmax, cmin, cmax = bounding_boxes[:,i]
            mask = np.zeros((oh_labels.shape[0], oh_labels.shape[1], 3))
            mask[:,:,0] = mask[:,:,1] = mask [:,:,2] = oh_labels[:,:,i]
            mask = mask.astype('uint8')

            chromosome = np.copy(img)
            chromosome = chromosome * mask + 255 * (1-mask)
            chromosome = chromosome[rmin:rmax, cmin:cmax]
            chromosome = pad_chromosome(chromosome, max_x, max_y)

            chrom_num = 12
            fn = "{0}_{1}_{2}.{3}".format(tags[i], str(chrom_num), fname, ftype)
            file_path = save_path + fn
            cv.imwrite(file_path, chromosome)
    else:
        for i in range(len(tags)*2):
            rmin, rmax, cmin, cmax = bounding_boxes[:,i]
            mask = np.zeros((oh_labels.shape[0], oh_labels.shape[1], 3))
            mask[:,:,0] = mask[:,:,1] = mask [:,:,2] = oh_labels[:,:,i]
            mask = mask.astype('uint8')
            
            chromosome = np.copy(img)
            chromosome = chromosome * mask + 255 * (1-mask)
            chromosome = chromosome[rmin:rmax, cmin:cmax]
            chromosome = pad_chromosome(chromosome, max_x, max_y)

            chrom_num = (i % 2) + 1
            tag_num = floor(i/2)
            fn = "{0}_{1}_{2}.{3}".format(tags[tag_num], str(chrom_num), fname, ftype)
            file_path = save_path + fn
            cv.imwrite(file_path, chromosome)

def get_max_bounding_box_dims(bounding_boxes):
    """ Get the maximum bounding box dimensions for a set of bounding boxes.

    Arguments:
        bounding_boxes: Numpy matrix containing the bounding box parameters.

    Returns:
        The biggest bounding box across both dimensions
    """
    max_x = 0
    max_y = 0

    for i in range(bounding_boxes.shape[1]):
        #[rmin, rmax, cmin, cmax]

        box = bounding_boxes[:,i]
        box_x = box[3] - box[2]
        box_y = box[1] - box[0]

        if box_x > max_x:
            max_x = box_x

        if box_y > max_y:
            max_y = box_y
    
    return max_x, max_y

def pad_chromosome(chromosome, max_x, max_y):
    """ Pads a chromosome images

    Arguments:
        chromosome: the chromosome iamge
        max_x: maximum x dimension
        max_y: maximum y dimenion

    Returns:
        The padded chromosome
    """
    chrom_x = chromosome.shape[1]
    chrom_y = chromosome.shape[0]

    pad_x = (max_x - chrom_x) / 2
    pad_y = (max_y - chrom_y) / 2

    x1, x2 = int(np.floor(pad_x)), int(np.ceil(pad_x))
    y1, y2 = int(np.floor(pad_y)), int(np.ceil(pad_y))

    padded_chromosome = np.pad(chromosome, [(y1,y2), (x1,x2), (0,0)], 'constant', constant_values=(255))

    return padded_chromosome

def fail_save(oh_labels, centroids, path):
    """ Handler for images that have != 46 chromosomes, 

    Arguments:
        oh_labels: matrix of one hot connected component labels
        centroids: oh label centroids
        path: path to image
    """
    print("  46 Chromosome have have to be detected, but {0} were detected. " \
            "This could also be due to noise or letters in the image.".format(one_hot_labels.shape[2]))
    visualise_components(oh_labels, show=False, save_path=path, centroids=centroids)

def visualise_components(labels, show=True, save_path=None, centroids=None, bounding_boxes=None):
    """ Visualisation helper for intermediate results

    Arguments:
        labels: one hot encoded connected component labels
        show: bool, save or show
        save_path: save path
        centroids: oh label centroids
        bounding boxes: bounding boxes of each one hot encoded label

    """

    bkg = np.sum(labels, axis=-1) == 0
    bkg = np.expand_dims(bkg, axis=-1)

    labels = np.concatenate([bkg, labels], axis=-1)
    labels = np.argmax(labels, axis=-1)
    labels = np.uint8(255*labels/np.max(labels))

    blank = 255 * np.ones_like(labels)
    img = cv.merge([labels, blank, blank])
    
    # cvt to BGR for display
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    # set bg label to black
    img = img * ~bkg

    plt.imshow(img)

    if centroids is not None:
        plt.plot(centroids[0,:], centroids[1,:], 'w+')

    if bounding_boxes is not None:
        # [rmin, rmax, cmin, cmax]
        # ymin 0  ymax 1 xmin 2 xmax 3
        y = bounding_boxes[0,:]
        x = bounding_boxes[2,:]
        h = bounding_boxes[1,:] - bounding_boxes[0,:]
        w = bounding_boxes[3,:] - bounding_boxes[2,:]
        # plt.plot(bounding_boxes[0,:], bounding_boxes[2,:], 'r+')
        ax = plt.gca()
        for i in range(bounding_boxes.shape[1]):
            rect = patches.Rectangle((x[i], y[i]), w[i], h[i], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    if save_path is not None:
        save_path = save_path.replace('jfif', 'jpg') # jfif not supported...
        plt.savefig(save_path)
    
    if show:
        plt.show()

    plt.close()

def process_chromosome(path, file_name, save_path, fail_path, pair=False, min_area=0.0003):
    """ Saves extracted chromosomes from a karyogram image.

    Arguments:
        path: path to karyogram
        file_name: file name of karyogram
        save_path: where the extracted chromosomes shall be saved
        fail_path: where the failed extractions visualisations will be save
        pair: optional, bool, whether the chromosomes of one class shall be extracted in a paired manner
        min_area: minimum pixel area of a chromosome (otherwise rejected)
    """
    num_labels, labels_im = get_labels(path)
    one_hot_labels = make_one_hot_labels(num_labels, labels_im)
    one_hot_labels = filter_labels(one_hot_labels, min_area)
    
    centroids = extract_label_centroids(one_hot_labels)
    
    if one_hot_labels.shape[2] != 46:
        f_path = os.path.join(fail_path, file_name)
        fail_save(one_hot_labels, centroids, f_path)
        return
    
    oh_labels = order_components(one_hot_labels, centroids) # order
    if pair:
        oh_labels = pair_components(oh_labels) # pair them up
    bounding_boxes = extract_bounding_boxes(oh_labels)

    extract_and_save_chromsomes(path, bounding_boxes, oh_labels, pair, save_path)

def process_chromosomes(img_path, save_path, fail_path, pair=False, min_area=0.0003):
    """ Saves extracted chromosomes from a karyogram image.

    Arguments:
        path: path to karyogram
        save_path: where the extracted chromosomes shall be saved
        fail_path: where the failed extractions visualisations will be save
        pair: optional, bool, whether the chromosomes of one class shall be extracted in a paired manner
        min_area: minimum pixel area of a chromosome (otherwise rejected)
    """
    file_names = os.listdir(img_path)
    for name in file_names:
        print("Processing: " + str(name))
        file_path = os.path.join(img_path, name)
        process_chromosome(file_path, name, save_path, fail_path, pair, min_area)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts chromosome from Karyogram image')
    parser.add_argument('-s', '--source', help='input folder', required=True)
    parser.add_argument('-d', '--dest', help='destination folder', required=True)
    parser.add_argument('-f', '--fails', help='folder for failed images', required=True)
    parser.add_argument('--pair', help='Whether chromosome pairs should be extracted', type=bool, nargs='?', default=False)
    parser.add_argument('--min_area', help='Minimum density of a chromosome in pixel percentage', type=float, nargs='?', default=0.0003)

    args = parser.parse_args()
    img_path = args.source
    save_path = args.dest
    fail_path = args.fails
    pair = args.pair
    min_area = args.min_area

    process_chromosomes(img_path, save_path, fail_path, pair, min_area)
