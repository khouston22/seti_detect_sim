import numpy as np

def db(x):
    """ Convert linear value to dB value, excludes x as list """
    return 10.*np.log10(abs(x)+1e-20)

def rms(x,axis=None):
    """ root-mean-square of a vector or rows/columns of a matrix """
    return np.sqrt(np.mean(np.square(abs(x)),axis))

def length(x):
    """ length of a scalar or list or np vector or matrix (product of dimensions) """
    xdims = np.shape(x)
    if len(xdims) == 0:
        return 1
    else:
        return np.prod(xdims)

def make_ndarray(x):
    """ make x into an indexable ndarray from scalar or list or np vector or matrix """
    if (isinstance(x, list)):
        return np.array(x)
    elif (type(x)!=np.ndarray):
        return np.array([x])
    else:
        return x


# temp = [[1, 2, 3, 4],[5,6,7,8]]
# temp = [1, 2, 3]
# temp = [1]
# # temp = np.array(temp)
# temp = 1
# # temp = []
# print(type(temp))
# temp = make_ndarray(temp)
# # if (isinstance(temp, list)):
# #     temp = np.array(temp)
# # elif (type(temp)!=np.ndarray):
# #     temp = np.array([temp])
# print(temp)
# print(f'{length(temp)=}')
# temp[0]

