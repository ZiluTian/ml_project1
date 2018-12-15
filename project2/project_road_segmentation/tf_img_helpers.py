import os 
from PIL import Image
import matplotlib.image as mpimg
from tf_global_vars import * 

def img_crop(im, w, h, border=0):
    """ Crop an image into 'patches'.
        @param im : The image to crop (array).
        @param w : width of a patch.
        @param h : height of a patch.
    """
    list_patches = []
    img_width = im.shape[0]
    img_height = im.shape[1]
    
    try: 
        img_channel = im.shape[2]
        if border != 0:
            im = numpy.array([numpy.pad(im[:, :, i], ((border, border), (border, border)), 'symmetric').T for i in range(img_channel)]).T
    except IndexError: 
        if border != 0:
            im = numpy.array([numpy.pad(im[:, :], ((border, border), (border, border)), 'symmetric').T]).T
            
    for i in range(0, img_height, h):
        for j in range(0, img_width, w):
            im_patch = im[j:j + w + 2 * border, i:i + h + 2 * border]
            list_patches.append(im_patch)
    return list_patches

def normalize_img(img):
    img[:,:,:,0] = (img[:,:,:,0] - numpy.mean(img[:,:,:,0]))/numpy.std(img[:,:,:,0])
    img[:,:,:,1] = (img[:,:,:,1] - numpy.mean(img[:,:,:,1]))/numpy.std(img[:,:,:,1])
    img[:,:,:,2] = (img[:,:,:,2] - numpy.mean(img[:,:,:,2]))/numpy.std(img[:,:,:,2])
    return img

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
#     if (numpy.max(rimg)!=0): 
#         rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
#     else: 
#         rimg = rimg.astype(numpy.uint8)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def extract_data_labels(filename, gt_filename, n_train, train_per, border):
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

    perm_index = numpy.random.permutation(range(n_train))

    imgs_train = [imgs[i] for i in perm_index[:int(n_train*train_per)]]
    if train_per < 1:
        imgs_val = [imgs[i] for i in perm_index[int(n_train*train_per):]]
    else:
        imgs_val = imgs_train

    num_imgs = len(imgs_train)

    img_patches = [img_crop(imgs_train[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, border) for i in range(num_imgs)]

    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    img_patches_val = [img_crop(imgs_val[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, border) for i in range(len(imgs_val))]
    data_val = [img_patches_val[i][j] for i in range(len(img_patches_val)) for j in range(len(img_patches_val[i]))]

    labels_train, labels_val = extract_labels(gt_filename, n_train, train_per, 0, perm_index)
    return (normalize_img(numpy.asarray(data)), labels_train, normalize_img(numpy.asarray(data_val)), labels_val)



# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]
    
    
# Extract label images
def extract_labels(filename, n_train, train_per, border, perm_index):
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

    gt_imgs_train = [gt_imgs[i] for i in perm_index[:int(n_train*train_per)]]
    if train_per < 1:
        gt_imgs_val = [gt_imgs[i] for i in perm_index[int(n_train*train_per):]]
    else:
        gt_imgs_val = gt_imgs_train

    gt_patches_train = [img_crop(gt_imgs_train[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, border) for i in range(len(gt_imgs_train))]
    data_train = numpy.asarray([gt_patches_train[i][j] for i in range(len(gt_patches_train)) for j in range(len(gt_patches_train[i]))])
    labels_train = numpy.asarray([value_to_class(numpy.mean(data_train[i])) for i in range(len(data_train))])

    gt_patches_val = [img_crop(gt_imgs_val[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, border) for i in range(len(gt_imgs_val))]
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
                l = 0
            else:
                l = 1
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


# Make an image summary for 4d tensor image with index idx
def get_image_summary(img, idx = 0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(V)
    V = V - min_value
    max_value = tf.reduce_max(V)
    V = V / (max_value*PIXEL_DEPTH)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V

# Make an image summary for 3d tensor image with index idx
def get_image_summary_3d(img):
    V = tf.slice(img, (0, 0, 0), (1, -1, -1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V


