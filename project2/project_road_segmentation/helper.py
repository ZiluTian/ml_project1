import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from sklearn.metrics import classification_report 

patch_size = 8

def load_image(infilename):
    """Read an image from a file to an array."""
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    """Convert image to 255 scale."""
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    """Concatenate an image and its groundtruth."""
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    """Crop image in patches of width w and height h."""
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

def extract_features(img):
    """Extract 6-dimensional features consisting of average RGB color as well as variance."""
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_2d(img):
    """Extract 2-dimensional features consisting of average gray color as well as variance."""
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

def extract_img_features(filename, dim = '2d'):
    """Extract features for a given image."""
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    if dim == '2d':
        X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    else:
        X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])

    return X

def value_to_class(v, foreground_threshold):
    """Classify patch to a label according to threshold."""
    df = np.mean(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def label_to_img(imgwidth, imgheight, w, h, labels):
    """Create image from label vector."""
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    """Overlap image and predicted-label image."""
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def print_eval(Y_train, Z_train, Y_test, Z_test): 
    print("Training set evaluation")
    print(classification_report(Y_train, Z_train, labels=[1]))

    print("Testing set evaluation")
    print(classification_report(Y_test, Z_test, labels = [1]))
    return 

def create_feature_vectors(feat_img, w, h):
    """Create a feature vector of dimension w x h for each patch. Input feat_img is the image resulting of a feature extraction algorithm."""
    feat_patches = img_crop(feat_img, w, h)
    feat_vec = np.array([feat_patches[i].ravel() for i in range(len(feat_patches))])
    return feat_vec

