from scipy import misc
import numpy as np
import blends
from matplotlib import pyplot as plt

def run_example( size = 64 ):
    """
    Run this file and result will be saved as 'rsult.jpg'
    Buttle neck: map 
    """
    modes = """
        normal
        add            substract      multiply       divide     
        dissolve       overlay        screen         pin_light
        linear_light   soft_light     vivid_light    hard_light    
        linear_dodge   color_dodge    linear_burn    color_burn
        light_only     dark_only      lighten        darken    
        lighter_color  darker_color   
        """
    top = misc.imresize( misc.imread('top.png')[:,:,:-1], (size,size,3) )
    base = misc.imresize( misc.imread('base.png')[:,:,:-1], (size,size,3) )
    modes = modes.split()
    num_of_mode = len( modes )
    result = np.zeros( [ size*2, size*(num_of_mode//2+2), 3 ])
    result[:size:,:size:,:] = top
    result[size:size*2:,:size:,:] = base
    for index in xrange( num_of_mode ):
        y = index // 2 + 1
        x = index % 2
        tmp= blends.blend( top, base, modes[index] )
        result[ x*size:(x+1)*size, y*size:(y+1)*size, : ] = tmp 
    # random blend
    result[-size::,-size::,:] = blends.random_blend( top, base )
    misc.imsave('result.jpg',result)

if __name__ == '__main__':
    run_example()