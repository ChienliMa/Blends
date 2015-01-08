#Blends:
    Light-weight and esay-to-use image blending tools in python
    
![example](https://raw.githubusercontent.com/ChienliMa/Blends/master/imgs/result.jpg)

--------------------

##Incentive:
    It's hard to find a image blending tool in python. Some rare tool either have little mode to choose
    or out_of_dated and hard to install. Therefore I just implement all methods available on in Internet.
##Recommended usage:

```python
import blends
result1 = blends.blend( top, base, 'mode' )
result2 = blends.blend( top, base, ['mode_1', ... ,'mode_n'])
```

##Not recommended usage:
```python 
import blends
result1 = blends.mode_name( top, base )
```
##Reference:
    MSDN:
        http://msdn.microsoft.com/en-us/library/hh706313.aspx
    WikiPeida:
        http://en.wikipedia.org/wiki/Blend_modes
        http://en.wikipedia.org/wiki/HSL_and_HSV
##Todo:
    Replace lambda with np.where and find if efficiency improve
