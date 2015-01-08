"""
Implement different Image blend Mode.

Author: Chienli Ma
Date: 2015.01.08

Incentive:
    It's hard to find a image blending tool in python.
    Some rare tool either have little mode or out_of_dated and hard to install. 
    Therefore I just implement all methods available on in Internet.


Recommended usage:
    import blends
    result1 = blends.blend( top, base, 'mode' )
    result2 = blends.blend( top, base, ['mode_1', ... ,'mode_n'])

Not recommended usage:
    import blends
    result1 = blends.mode_name( top, base )

Reference:
    MSDN:
        http://msdn.microsoft.com/en-us/library/hh706313.aspx

    WikiPeida:
        http://en.wikipedia.org/wiki/Blend_modes
        http://en.wikipedia.org/wiki/HSL_and_HSV

Todo:
    Replace lambda with np.where and find if efficiency improve
"""
import numpy as np
import pdb

###########
# lambdas #
###########
darker_color_func = lambda x,y,m,n: x if m<n else y
lighter_color_func = lambda x,y,m,n: x if m>n else y
dissolve_func = lambda x, y, m, n : x/m if m>0 and m>n else y
overlay_func = lambda x,y: 2*x*y if x <0.5 else 1- 2*(1-x)*(1-y)
hard_light_func = lambda x,y: 2*x*y if x<0.5 else 1 - 2*(1-x)*(1-y)
vivid_light_func = lambda x,y: (2*x+y-1)/(2*x+10e-5) if x<0.5 else y/(2*(1-x+10e-5))
pin_light_func = lambda x,y: max(x,2*y-1) if x>0.5 else min(x,2*y) 
color_dodge_func = lambda x,y: np.clip( x/(1-y) if y<1 else 1, 0 , 1 )
color_burn_func = lambda x,y: 1-(1-x)/y if y>0 else 0



def blend( top, base, mode, top_alpha = None, base_alpha = None, mask = None ):
    """
    Blend top to base using specific mode

    Parameters:
        top - 3 dimension numpy.ndarray with 3 channels are [ width, height, \
                channel ]
        base - same as top, has same shape as top
        mode - string of mode name
        top_alpha - 3 dimension numpy.ndarray, the last channel represents \
                    the alpha channel and should have same value
        base_alpha -  same as top_alpha
        mask - 3 dimension numpy.ndarray, where mask != 0 will be blended

    Returns:
        3 dimension numpy.ndarray with same shape as top and base
    """
    assert top.dtype == base.dtype
    assert top.shape == base.shape

    w, h, c = top.shape
    mode = globals().get( mode )
    mask = ( mask or np.ones( top.shape ) ).flatten()
    if top.dtype == np.float: # if is float
        # flatten array for 'map' opeartion
        top = top.flatten()
        result = result.flatten()
        if top_alpha != None:
            top_alpha = top_alpha.flatten() 
        if base_alpha != None:    
            base_alpha = base_alpha.flatten() 
        result = copy( mode(top, base, top_alpha, base_alpha), base, mask)
    else:
        # flatten array for 'map' opeartion
        top = top.flatten() / 255.0
        base = base.flatten() / 255.0      
        if top_alpha != None:
            top_alpha = top_alpha.flatten()/255.0 
        if base_alpha != None:
            base_alpha = base_alpha.flatten()/255.0 
        result = copy( mode(top, base, top_alpha, base_alpha), base, mask)
        result = (result*255).astype('uint8')

    return result.reshape( [w, h, c] )

def random_blend( top, base, top_alpha = None , base_alpha = None, modes = None ):
    """
    Blend top to base using random mode. Parameters are similar to blend.

    Parameters:
    modes - List of mode names,Optional. This function will randomly pick \
            a mode in modes to blend images.  If it is not specified, this \
            method will pick a mode in all available modes.
    """
    if modes == None:
        modes = all_modes()
    modes = modes.split()
    mode = modes[np.random.randint(0,len(modes))]
    return blend( top, base, mode , top_alpha = None, base_alpha = None)

########################
# Single blend methods #
########################
def normal( top, base, top_alpha = None, base_alpha = None ):
    """
    Normal mode simply replace base layer with top layer
    """
    return base

def darken( top, base, top_alpha = None, base_alpha = None ):
    """
    Darken it. = =( yes, I need a better explaination here.)
    """
    if top_alpha == None:
        top_alpha = 1
    if base_alpha == None:
        base_alpha = 1
    return np.minimum( (1-top_alpha)*base+top, (1-base_alpha)*top+base )

def lighten( top, base, top_alpha = None, base_alpha = None ):
    """
    Lighten it. = =( yes, I need a better explaination here.)
    P.S.: 
    In MSDN's Document, darken and lighten are the same, here I use
    maximum instead of minimum in the darken
    """
    if top_alpha == None:
        top_alpha = 1
    if base_alpha == None:
        base_alpha = 1    
    tmp = np.maximum( (1-top_alpha)*base+top, (1-base_alpha)*top+base )
    return np.clip( tmp, 0, 1 )

def darker_color( top, base, top_aplha = None, base_alpha = None ):
    """
    Darker color mode simply take the pixel with lower luminance
    """
    top_lum = lum( top )
    base_lum = lum( base )
    return np.array( map(darker_color_func, base, top, base_lum, top_lum) )

def lighter_color( top, base, top_aplha = None, base_alpha = None ):
    """
    Lighter color mode simply take the pixel with bigger luminance
    """
    top_lum = lum( top )
    base_lum = lum( base )
    return np.array( map(lighter_color_func, base, top, base_lum, top_lum) )

def dissolve( top, base, top_alpha = None, base_alpha = None):
    """
    The dissolve mode takes random pixels from both layers. With high \
    opacity, most pixels are taken from the top layer. With low opacity \
    most pixels are taken from the bottom layer. 
    """
    if top_alpha == None:
        top_alpha = np.ones(top.size) 
    rand = np.random.uniform( 0, 1, top.size )
    return np.array( map( dissolve_func, top, base, top_alpha,rand ) )

def add( top, base, top_alpha = None, base_alpha = None ):
    """
    Add mode simply sums two layers and clips the result to [0,1] 
    """
    if top_alpha == None:
        top_alpha = 1
    if base_alpha == None:
        base_alpha = 1    
    return np.clip( top * top_alpha + base * base_alpha, 0, 1 )

def substract( top, base, top_alpha = None, base_alpha = None ):
    """
    Add mode simply substracts two layers and clips the result to [0,1] 
    """
    return np.clip( base - top , 0, 1 )

def multiply( top, base, top_alpha = None, base_alpha = None ):
    """
    Multiply mode simply multiply two layers
    """
    if top_alpha == None:
        top_alpha = 1
    if base_alpha == None:
        base_alpha = 1    
    return top*base + top*(1-base_alpha) + base*(1-top_alpha)

def divide( top, base, top_alpha = None, base_alpha = None ):
    """
    Divide modes simply divid base layer by top layer and clips it yo [0,1]
    Divide by black yield white
    """  
    return np.clip( base / (top+10e-7) , 0, 1 )

def screen( top, base, top_alpha = None, base_alpha = None ):
    """
    Screen blend mode inverts both layers, multiplies them, and then inverts \
    that result.
    """
    return 1 - ( 1 - top ) * ( 1 - base )

def overlay( top, base, top_alpha = None, base_alpha = None ):
    """
    Overlay combines Multiply and Screen blend modes.[3] The parts of the top \
    layer where base layer is light become lighter, the parts where the base \
    layer is dark become darker. 
    """
    return np.array( map( overlay_func, base, top ))

def hard_light( top, base, top_alpha = None, base_alpha = None ):
    """
    Hard light mode!  = =( yes, I need a better explaination here.)
    """
    return np.array( map( hard_light_func, top, base ) )

def soft_light( top, base, top_alpha = None, base_alpha = None ):
    """
    Soft light mode!  = =( yes, I need a better explaination here.)
    P.S.:
    There are many differet formular for implemnt soft light mode.
    Here I Pegtop's formular, for more details look at loft light mode in: 
            http://en.wikipedia.org/wiki/Blend_modes#Soft_Light
    """
    return ( 1 - 2*top )*base**2 + 2*base*top

def vivid_light( top, base, top_alpha = None, base_alpha = None ):
    """
    Vivid Light: 
    This blend mode combines Color Dodge and Color Burn \
    (rescaled so that neutral colors become middle gray). Dodge applies \
    when values in the top layer are lighter than middle gray, and burn \
    to darker values. 

    P.S.:
    For a simpler implementation, I use small sigma to avoid top of being \
    0 or 1. So there might be a small diferece between actual result and \
    expected result. 
    """
    return np.clip( map( vivid_light_func, top, base ), 0 ,1 )

def linear_light( top, base, top_alpha = None, base_alpha = None ):
    """
    Linear Light: this blend mode combines Linear Dodge and Linear Burn \
    (rescaled so that neutral colors become middle gray). Dodge applies to \
    values of top layer lighter than middle gray, and burn to darker values. \
    The calculation simplifies to the sum of bottom layer and twice the top \
    layer, subtract 1.
    """
    return np.clip( base + 2 * top - 1, 0, 1)

def pin_light( top, base, top_alpha = None, base_alpha = None ):
    """
    Pin light mode!  = =( yes, I need a better explaination here.)
    """
    return np.clip( map( pin_light_func, base, top ), 0 ,1 )

def color_dodge( top, base, top_alpha = None, base_alpha = None ):
    """
    Color Dodge blend mode divides the bottom layer by the inverted top layer. 
    """
    return np.array( map( color_dodge_func, base, top ) )

def linear_dodge( top, base, top_alpha = None, base_alpha = None ):
    """
    Linear Dodge blend mode simply sums the values in the two layers. \
    Blending with white gives white.
    """
    return add( top, base )

def color_burn( top, base, top_alpha = None, base_alpha = None ):
    """
    Color Burn mode divides the inverted bottom layer by the top layer, 
    and then inverts the result. 
    """
    return np.clip( np.array( map( color_burn_func, base, top ) ), 0, 1 )

def linear_burn( top, base, top_alpha = None, base_alpha = None ):
    """
    Linear Burn mode sums the value in the two layers and subtracts 1
    """
    return np.clip( top+base-1, 0, 1 )

def dark_only( top, base, top_alpha = None, base_alpha = None ):
    """
    Pixel's value are given by (min(r1,r2), min(g1,g2), min(b1,b2))
    """
    return np.minimum( top, base)

def light_only( top, base, top_alpha = None, base_alpha = None ):
    """
    Pixel's value are given by (min(r1,r2), min(g1,g2), min(b1,b2))
    """
    return np.maximum( top, base )

# Auxilary function
def lum( src ):
    """
    Luminance = 0.5 * min(R,G,B) + 0.5 * max(R,G,B)
    input: image with channel of 3
    output: luminance map with channel of 3. Channels are the same.

    Reference:
    http://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
    """
    lum = np.array( [ src[0::3],
                      src[1::3],
                      src[2::3] ] )
    return lum.mean(axis=0).repeat(3)

def copy( src, dst, mask ):
    """
    Behave like cv.Copy. src, dst and mask should have exact same size.
    """
    assert src.shape == dst.shape
    assert src.shape == mask.shape
    result = dst.copy()
    result[ np.where( mask!=0 )] = src[ np.where( mask!=0 )]
    return result  

def help():
    print """
Recommand usages:
    blends.blend( top, base, mask, mode, *[top_alpha[base_alpha]])

Type methods to view all single methods.
    """

def all_modes():
    return """
    normal
    add            substract      multiply       divide     
    dissolve       overlay        screen         pin_light
    linear_light   soft_light     vivid_light    hard_light    
    linear_dodge   color_dodge    linear_burn    color_burn
    light_only     dark_only      lighten        darken    
    lighter_color  darker_color               
    """