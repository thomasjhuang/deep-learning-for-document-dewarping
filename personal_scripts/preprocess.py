import numpy as np
import os, sys, cv2, shutil, argparse, time, concurrent.futures, imageio
import warnings
import itertools as it
from PIL import Image
from random import shuffle
from math import floor
from skimage import data
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.batches import Batch

def quad_as_rect(quad):
    if quad[0] != quad[2]: return False
    if quad[0] != quad[2]: return False
    if quad[1] != quad[7]: return False
    if quad[4] != quad[6]: return False
    if quad[3] != quad[5]: return False
    return True

def quad_to_rect(quad):
    assert(len(quad) == 8)
    assert(quad_as_rect(quad))
    return (quad[0], quad[1], quad[4], quad[3])

def rect_to_quad(rect):
    assert(len(rect) == 4)
    return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])

def shape_to_rect(shape):
    assert(len(shape) == 2)
    return (0, 0, shape[0], shape[1])

def griddify(rect, w_div, h_div):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    x_step = w / float(w_div)
    y_step = h / float(h_div)
    y = rect[1]
    grid_vertex_matrix = []
    for _ in range(h_div + 1):
        grid_vertex_matrix.append([])
        x = rect[0]
        for _ in range(w_div + 1):
            grid_vertex_matrix[-1].append([int(x), int(y)])
            x += x_step
        y += y_step
    grid = np.array(grid_vertex_matrix)
    return grid

def distort_grid(org_grid, max_shift):
    new_grid = np.copy(org_grid)
    x_min = np.min(new_grid[:, :, 0])
    y_min = np.min(new_grid[:, :, 1])
    x_max = np.max(new_grid[:, :, 0])
    y_max = np.max(new_grid[:, :, 1])
    new_grid += np.random.randint(- max_shift, max_shift + 1, new_grid.shape)
    new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
    new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
    return new_grid

def grid_to_mesh(src_grid, dst_grid):
    assert(src_grid.shape == dst_grid.shape)
    mesh = []
    for i in range(src_grid.shape[0] - 1):
        for j in range(src_grid.shape[1] - 1):
            src_quad = [src_grid[i    , j    , 0], src_grid[i    , j    , 1],
                    src_grid[i + 1, j    , 0], src_grid[i + 1, j    , 1],
                    src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
                    src_grid[i    , j + 1, 0], src_grid[i    , j + 1, 1]]
            dst_quad = [dst_grid[i    , j    , 0], dst_grid[i    , j    , 1],
                    dst_grid[i + 1, j    , 0], dst_grid[i + 1, j    , 1],
                    dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
                    dst_grid[i    , j + 1, 0], dst_grid[i    , j + 1, 1]]
            dst_rect = quad_to_rect(dst_quad)
            mesh.append([dst_rect, src_quad])
    return mesh

def resize(args):
    img_size,filepath = args
    sq_img = cv2.imread(filepath) # square image
    scaled_sq_img = resizeAndPad(sq_img, (img_size,img_size), 127)
    cv2.imwrite(filepath, scaled_sq_img)

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC
    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.relpath(datadir))
    data_files = list(filter(lambda file: file.endswith('.png'), all_files))
    return data_files

def randomize_files(file_list):
    shuffle(file_list)
    return file_list

def get_training_and_testing_sets(train, file_list):
    #Initial split for training and test data
    split = train
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    final_testing = file_list[split_index:]
    #Secondary split for validation
    split_index = floor(len(training) * split)
    final_training = training[:split_index]
    return final_training, final_testing

def warp(args):
    filename, root, fold_A = args
    im = Image.open(os.path.join(root,filename))
    dst_grid = griddify(shape_to_rect(im.size), 4, 4)
    src_grid = distort_grid(dst_grid, 50)
    mesh = grid_to_mesh(src_grid, dst_grid)
    im = im.transform(im.size, Image.MESH, mesh)
    im.save(os.path.join(fold_A,root.rsplit('/', 1)[-1],filename))

def png2jpg(args):
    filepath = args
    im = Image.open(filepath)
    im = im.convert('RGB')
    im.save(os.path.splitext(filepath)[0] + '.jpg', quality=100)


def rotate_image(mat, angle):

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def imgaug(args):
    # Number of batches and batch size for this example
    filename, root, fold_A = args
    img = cv2.imread(os.path.join(root,filename))
    print('image opened ' + os.path.join(root,filename))
    batch_size = 4
    for i in range(0,batch_size):
        imageio.imwrite(os.path.join(root, os.path.splitext(filename)[0] + '_' + str(i) + '.jpg'), img) #convert the current image in B into a jpg from png
    nb_batches = 1

    # Example augmentation sequence to run in the background
    sometimes = lambda aug: iaa.Sometimes(0.4, aug)
    augseq = iaa.Sequential(
            [
                iaa.PiecewiseAffine(scale=(0.01, 0.01005))
            ]
        )

    # Make batches out of the example image (here: 10 batches, each 32 times
    # the example image)
    batches = []
    for _ in range(nb_batches):
        batches.append(Batch(images=[img] * batch_size))

    #Save images
    for batch in augseq.augment_batches(batches, background=False):
        count = 0
        for img in batch.images_aug:
            path = os.path.join(fold_A,root.rsplit('/', 1)[-1], os.path.splitext(filename)[0] + '_' + str(count) + '.jpg')
            cv2.imwrite(path, img)
            print('image saved as: ' + path)
            count +=1

def resize_and_rotate(filename):
    img = cv2.imread(filename)[:, :, ::-1]
    print('image opened: ' + filename)
    img = rotate_image(img,90)
    img = ia.imresize_single_image(img, (1024, 2048))

    cv2.imwrite(filename, img)
    print('image rotated and resized saved as: ' + filename)



def upsample(args):
    filename, root = args
    img = imageio.imread(os.path.join(root, filename))
    img = rotate_image(img,-90)
    img = ia.imresize_single_image(img, (4400, 3400))
    cv2.imwrite(os.path.join(root, filename))


def main():
    #Note: Atm, it is necessary to have a 'train', 'val', and 'test' folder under folder A and B already before running the script
    #arg parsing
    parser = argparse.ArgumentParser('Completely preprocess all data for pix2pix')
    parser.add_argument('--raw_data', dest='raw_data', help='input directory for all initial flatbed images', type=str, default='./raw_data')
    parser.add_argument('--dest_dir', dest='dest_dir', help='destination directory for processed images', type=str, default='./datasets')
    parser.add_argument('--train', dest='train', help='% of data that are training (this will also determine validation split)', type=float, default=0.7)
    parser.add_argument('--imgsize', dest='imgsize', help='# of pixels (in an n pixels x n pixels square) that you want images to be resized to', type=int, default=512)
    parser.add_argument('--split', dest='split', help='determine if you want to split or not', type=bool, default=False)
    parser.add_argument('--resize', dest='resize', help='to resize a set of images', type=bool, default=False)
    parser.add_argument('--resize_fold', dest='resize_fold', help='the folder where you want jpgs to be resized', type=str, default='./')
    parser.add_argument('--preprocess', dest='preprocess', help='complete preprocessing', type=bool, default=False)
    parser.add_argument('--removepng', dest='removepng', help='removes pngs', type=bool, default=False)
    parser.add_argument('--png2jpg', dest='png2jpg', help='convert folder of choice to jpg', type=str, default="")


    args = parser.parse_args()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    #print args
    for arg in vars(args):
        print('[%s] = ' % arg, getattr(args, arg))

    start = time.clock()
    total = time.clock()

    #1. Begin splitting data
    if(args.split == True):
        print('Splitting data...')
        datadir = args.raw_data
        data_files = get_file_list_from_dir(datadir)
        randomized = randomize_files(data_files)
        training, testing = get_training_and_testing_sets(args.train, randomized)

        train_A_dir = os.path.join(args.dest_dir,'train_A')
        train_B_dir = os.path.join(args.dest_dir,'train_B'))
        test_A_dir = os.path.join(args.dest_dir,'test_A'))
        test_B_dir = os.path.join(args.dest_dir,'test_B'))

        os.mkdir(train_A_dir)
        os.mkdir(train_B_dir)
        os.mkdir(test_A_dir)
        os.mkdir(test_B_dir)

        count = 0
        for f in training:
            shutil.copy(args.raw_data + f, os.path.join( train_B_dir, "IMG_" + str(count) + ".png")
            count += 1
        for f in testing:
            shutil.copy(args.raw_data + f, os.path.join( test_B_dir, "IMG_" + str(count) + ".png")
            count +=1
        print('Splitting took {:.3f} seconds'.format((time.clock() - start)*1000.0))

    if(args.preprocess == True):
        #2. Perform image warping and save those image into folder A
        print('Resizing, Warping, and Converting images...')
        total2 = time.clock()

        #imgaug
        start = time.clock()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for root, dirs, files in os.walk():
                conv_list = ((file, root, args.fold_A) for file in files)
                executor.map(imgaug, conv_list)
        executor.shutdown(wait=True)
        print('Augmentation took {:.3f} seconds'.format((time.clock() - start)*1000.0))

    if(args.png2jpg != ""):
        print("Converting all pngs to jpgs")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for root, dirs, files in os.walk(args.png2jpg):
                conv_list = (os.path.join(root,file) for file in files)
                executor.map(png2jpg, conv_list)
        executor.shutdown(wait=True)
        print("Conversion to jpgs done")

    if(args.resize == True):
        print('Resizing data')
        start = time.clock()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            process_list = []
            for filename in os.listdir(args.resize_fold):
                if filename.endswith(".jpg"):
                    process_list.append(args.resize_fold + filename)
            executor.map(resize_and_rotate, process_list)
        executor.shutdown(wait=True)
        print('Rotations took {:.3f} seconds'.format((time.clock() - start)*1000.0))

    if(args.removepng == True):
        #3. Removal of pngs
        print("Removing png's")
        for root, dirs, files in os.walk(args.fold_A):
            for file in files:
                if file.endswith('.png'):
                    os.remove(os.path.join(root,file))
        for root, dirs, files in os.walk(args.fold_B):
            for file in files:
                if file.endswith('.png'):
                    os.remove(os.path.join(root,file))
        print("Png's removed")

if __name__ == "__main__":
    main()
