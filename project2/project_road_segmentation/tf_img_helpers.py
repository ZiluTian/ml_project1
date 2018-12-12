# Helper functions for image preprocessing 
import numpy 
import matplotlib.image as mpimg
import os 

from tf_global_vars import * 

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def normalize_img(img):
    img[:,:,:,0] = (img[:,:,:,0] - numpy.mean(img[:,:,:,0]))/numpy.std(img[:,:,:,0])
    img[:,:,:,1] = (img[:,:,:,1] - numpy.mean(img[:,:,:,1]))/numpy.std(img[:,:,:,1])
    img[:,:,:,2] = (img[:,:,:,2] - numpy.mean(img[:,:,:,2]))/numpy.std(img[:,:,:,2])
    return img

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def extract_data_labels(filename, gt_filename, n_train, train_per):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, n_train + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    imgs_train = imgs[:int(n_train*train_per)]
    if train_per<1:
        imgs_val = imgs[int(n_train*train_per):]
    else:
        imgs_val = imgs_train
    IMG_WIDTH = imgs_train[0].shape[0]
    IMG_HEIGHT = imgs_train[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs_train[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(len(imgs_train))]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    img_patches_val = [img_crop(imgs_val[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(len(imgs_val))]
    data_val = [img_patches_val[i][j] for i in range(len(img_patches_val)) for j in range(len(img_patches_val[i]))]

    labels_train, labels_val = extract_labels(gt_filename, n_train, train_per)
    return (normalize_img(numpy.asarray(data)), labels_train, normalize_img(numpy.asarray(data_val)), labels_val)


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [1, 0]
    else:
        return [0, 1]
    
    
# Extract label images
def extract_labels(filename, n_train, train_per):
    """Extract the labels into a 1-hot matrix [image index, label index]."""

    gt_imgs = []
    for i in range(1, n_train+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    gt_imgs_train = gt_imgs[:int(n_train*train_per)]
    if train_per<1:
        gt_imgs_val= gt_imgs[int(n_train*train_per):]
    else:
        gt_imgs_val = gt_imgs_train

    gt_patches_train = [img_crop(gt_imgs_train[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(len(gt_imgs_train))]
    data_train = numpy.asarray([gt_patches_train[i][j] for i in range(len(gt_patches_train)) for j in range(len(gt_patches_train[i]))])
    labels_train = numpy.asarray([value_to_class(numpy.mean(data_train[i])) for i in range(len(data_train))])

    gt_patches_val = [img_crop(gt_imgs_val[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(len(gt_imgs_val))]
    data_val = numpy.asarray([gt_patches_val[i][j] for i in range(len(gt_patches_val)) for j in range(len(gt_patches_val[i]))])
    labels_val = numpy.asarray([value_to_class(numpy.mean(data_val[i])) for i in range(len(data_val))])

    # Convert to dense 1-hot representation.
    return (labels_train.astype(numpy.float32), labels_val.astype(numpy.float32))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img    