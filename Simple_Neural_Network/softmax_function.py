Python 3.8.1 (tags/v3.8.1:1b293b6, Dec 18 2019, 22:39:24) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> x = np.array([2.0,1.0,0.1])
>>> exp_x = np.exp(x)
>>> exp_x
array([7.3890561 , 2.71828183, 1.10517092])
>>> sum_exp = sum(exp_x)
>>> sum_exp
11.212508845465344
>>> y = [i / sum_exp for i in exp_x]
>>> y
[0.6590011388859679, 0.2424329707047139, 0.09856589040931818]
>>> sum(y)
1.0
>>> np.argmax(y)
0
>>> import matplotlib.pyplot as plt
>>> plt.plot(x,sum(y))
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    plt.plot(x,sum(y))
  File "C:\Users\DEEPAK KUMAR RAI\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\pyplot.py", line 2787, in plot
    return gca().plot(
  File "C:\Users\DEEPAK KUMAR RAI\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\axes\_axes.py", line 1665, in plot
    lines = [*self._get_lines(*args, data=data, **kwargs)]
  File "C:\Users\DEEPAK KUMAR RAI\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\axes\_base.py", line 225, in __call__
    yield from self._plot_args(this, kwargs)
  File "C:\Users\DEEPAK KUMAR RAI\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\axes\_base.py", line 391, in _plot_args
    x, y = self._xy_from_xy(x, y)
  File "C:\Users\DEEPAK KUMAR RAI\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\axes\_base.py", line 269, in _xy_from_xy
    raise ValueError("x and y must have same first dimension, but "
ValueError: x and y must have same first dimension, but have shapes (3,) and (1,)
>>> 