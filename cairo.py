import argparse
import pprint

from PIL import Image # Pillow
import numpy as np 
import numba
from tqdm import tqdm

verbose = False

def vprint(*args, **kwargs):
    """
    Verbose print
    """

    if verbose:
        print(*args, **kwargs)

def timer(f):
    """
    Decorator function that times the execution time of a decorated function
    """

    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        vprint('{0} took {1:.2f}s'.format(f.__qualname__, time.time()-start))
        return result
    return wrapper

def _file(x):
    """
    raises an Exception if a file x does not exist
    """

    import os
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError(f'Input file {x} does not exist')
    return x

def gradient(im):
    """Reads an Image file and outputs an energy map / gradient representation of it
    
    Arguments:
        file {string} -- filepath
    
    Returns:
        PIL.Image -- an energy map representation of the input image
    """
    # energy map
    im_arr = np.asarray(im.convert('F'))
    height, width = im_arr.shape
    result = np.zeros_like(im_arr)
    for x in range(height):
        result[x,:] = im_arr[(x + 1) % height, :] - im_arr[( x - 1 ) % height,:]
    result = np.square(result)
    for y in range(width):
        result[:,y] += np.square(im_arr[:, ( y + 1 ) % width] - im_arr[:, ( y - 1 ) % width])
    result = np.sqrt(result)
    return Image.fromarray(result)

@timer
def preprocess_image(file):
    im = Image.open(file, 'r')
    return im, im.size[0], im.size[1]
    

@numba.jit
def vertical_seam(im):
    """ Finds a vertical path in a given (greyscale gradient) image of lowest energy"""
    gradient = np.transpose(np.asarray(im))
    width, height = im.size
    c = np.zeros(im.size)
    c[:,0] = gradient[:,0]
    for y in range(1, height):
        for x in range(width):
            min_c = c[x,y-1]
            if x > 0:
                if c[x-1,y-1] < min_c:
                    min_c = c[x-1,y-1]
            if x < width - 1:
                if c[x+1,y-1] < min_c:
                    min_c = c[x+1,y-1]
            c[x,y] = gradient[x,y] + min_c
    # path reconstruction
    path = []
    # minimum of last line
    min_x = np.argmin(c[:,-1])
    path.append(min_x)
    x = min_x
    for y in range(height-1, -1, -1):
        a = c[x-1,y-1]
        b = c[x,y-1]
        d = c[x+1,y-1]
        if a < b and a < d and x > 0:
            pred = x - 1
        elif b > d and x < width - 1:
            pred = x + 1
        else:
            pred = x
        x = pred
        if x < 0:
            print("--x NEGATIVE , ur fucked--")
        path.append(pred)
    return path

def remove_vertical_seam(im, path):
    original = im.load()
    width, height = im.size
    tighter = Image.new(im.mode, (width-1, height))
    removed_at_row = set()
    min_x = min(path)
    max_x = max(path)
    tighter.paste(im.crop((0,0,min_x,height)), (0,0))
    tighter.paste(im.crop((max_x,0,width,height)), (max_x-1,0))
    result = tighter.load()
    for x in range(min_x, max_x):
        for y in range(height):
            if path[height-y-1] != x and y not in removed_at_row:
                result[x,y] = original[x,y]
            elif path[height-y-1] == x:
                removed_at_row.add(y)
            else:
                result[x-1,y] = original[x,y]
    return tighter

@timer
def resize(im, width_to_remove):
    for _ in tqdm(range(width_to_remove)):
        path = vertical_seam(gradient(im))
        im = remove_vertical_seam(im, path)
    return im

def _transpose_image(im):
    return im.transpose(Image.TRANSPOSE)


def main():
    print('''
  ____      _           
 / ___|__ _(_)_ __ ___  
| |   / _` | | '__/ _ \ 
| |__| (_| | | | | (_) |
 \____\__,_|_|_|  \___/ 
''')
    parser = argparse.ArgumentParser(description='Context Aware Image Resizing in Python')
    parser.add_argument('size', nargs='*', help="the desired size, can either be one scaling factor (<= 1.0) or desired 'x y' dimensions", type=float)
    parser.add_argument('input', help='the desired image file to rescale', type=_file)
    parser.add_argument('output', help='output filename')
    parser.add_argument("-v", "--verbose", help="forces verbose output", action='store_true')
    args = parser.parse_args()
    if args.verbose:
        print('Verbose mode activated')
        global verbose
        verbose = True
    if len(args.size) > 2 or not len(args.size):
        raise argparse.ArgumentTypeError('Incorrent number of size specifications, only either 1 or 2 arguments allowed')
    im, width, height = preprocess_image(args.input)
    width, height = im.size
    desired_width, desired_height = width, height
    if len(args.size) == 2:
        if args.size[0] >= width or args.size[0] < 1 or args.size[1] >= height or args.size[1] < 1:
            raise argparse.ArgumentTypeError('Scaling values must be between 1 and original values')
        desired_width, desired_height = list(map(int, args.size))
    else:
        if args.size[0] > 1. or args.size[0] < .0:
            raise argparse.ArgumentTypeError('Scaling factor must be between 0 and 1')
        desired_width, desired_height = int(width * args.size[0]), int(height * args.size[0]) 
    print('Reducing width')
    im = resize(im, width-desired_width)
    im = _transpose_image(im)
    print('Reducing height')
    im = resize(im, height-desired_height)
    vprint(f'Saving to {args.output}')
    _transpose_image(im).convert('RGB').save(args.output)

if __name__ == '__main__':
    main()