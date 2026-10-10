from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

# New RGB/indexed-mask dataset helpers used by both notebooks and main.py.
from pathlib import Path
from PIL import Image
import tensorflow as tf


def prepare_dataset(dataset_path, limit=5):
    """Validate matching RGB images/masks and select a reproducible split."""
    dataset_path = Path(dataset_path)
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer or None")
    images = {x.stem: x for x in (dataset_path / 'data').glob('*.png')}
    masks = {x.stem: x for x in (dataset_path / 'mask').glob('*.png')}
    if not images or images.keys() != masks.keys():
        raise ValueError('Dataset must contain exactly matched data/*.png and mask/*.png')
    keys = sorted(images)
    if limit is not None:
        keys = sorted(keys, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
    if limit is not None:
        if limit > len(keys):
            raise ValueError('Requested limit exceeds available image/mask pairs')
        keys = keys[:limit]
    labels = set()
    for key in keys:
        with Image.open(images[key]) as im, Image.open(masks[key]) as mask:
            values = np.asarray(mask)
            if values.ndim != 2 or im.size != mask.size:
                raise ValueError(f'Invalid mask or mismatched dimensions: {key}')
            labels.update(np.unique(values).tolist())
    if labels != {0, 1, 2}:
        raise ValueError(f'Expected mask IDs 0,1,2, found {labels}')
    rng = np.random.default_rng(42)
    order = rng.permutation(len(keys))
    ntest = int(np.ceil(len(keys) * .2))
    nval = int(np.ceil((len(keys) - ntest) * .1))
    splits = {'test': order[:ntest], 'validation': order[ntest:ntest+nval], 'train': order[ntest+nval:]}
    if limit is not None:
        splits = {name: np.arange(len(keys)) for name in ('train', 'validation', 'test')}
    manifest = {name: [keys[i] for i in indices] for name, indices in splits.items()}
    return images, masks, manifest


def make_dataset(images, masks, ids, size=128, batch_size=4, training=False, seed=42):
    """Load float RGB images and unnormalized integer masks using tf.data."""
    if not ids:
        raise ValueError("Dataset split cannot be empty")
    def load(image_path, mask_path):
        image = tf.io.decode_png(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize(tf.cast(image, tf.float32), [size, size]) / 255.
        mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
        mask = tf.image.resize(mask, [size, size], method='nearest')
        return image, tf.cast(tf.squeeze(mask, -1), tf.int32)
    ds = tf.data.Dataset.from_tensor_slices(
        ([str(images[k]) for k in ids], [str(masks[k]) for k in ids]))
    if training:
        ds = ds.shuffle(len(ids), seed=seed)
    return ds.map(load, num_parallel_calls=2).batch(batch_size).prefetch(1)


def save_prediction(save_path, key, prediction):
    """Save class IDs and a separate color visualization without altering predictions."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(prediction)
    if prediction.ndim != 2 or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError("Prediction must be a 2D mask containing class IDs 0, 1, 2")
    prediction = prediction.astype('uint8')
    Image.fromarray(prediction).save(save_path / f'{key}_mask.png')
    palette = np.array([[0,0,0], [255,100,0], [0,200,80]], dtype='uint8')
    Image.fromarray(palette[prediction]).save(save_path / f'{key}_color.png')


def plot_predictions(images, masks, ids, pred_dir, output):
    """Return the image/ground-truth/prediction figure used by CLI and notebook."""
    import matplotlib.pyplot as plt
    ids = ids[:5]
    fig, axes = plt.subplots(len(ids), 3, figsize=(12, 3 * len(ids)), squeeze=False)
    for row, key in enumerate(ids):
        with Image.open(images[key]) as image:
            axes[row, 0].imshow(image.convert('RGB'))
        with Image.open(masks[key]) as mask:
            axes[row, 1].imshow(np.asarray(mask), cmap='viridis', vmin=0, vmax=2, interpolation='nearest')
        with Image.open(Path(pred_dir) / f'{key}_mask.png') as pred:
            axes[row, 2].imshow(np.asarray(pred), cmap='viridis', vmin=0, vmax=2, interpolation='nearest')
        for col, title in enumerate((f'Image {key}', 'Ground truth', 'Prediction')):
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
    fig.tight_layout()
    fig.savefig(Path(output) / 'prediction_comparison.png', dpi=140)
    return fig
