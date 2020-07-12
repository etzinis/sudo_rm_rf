"""!
@brief A general bar progress bar display container for all functions
applied on a list or or an enumerable structure of elements
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

from tqdm import tqdm
import numpy as np


def progress_bar_wrapper_old(func,
                             l,
                             message='Processing...'):
    from progress.bar import ChargingBar
    """
    !
    :param l: List of elements
    :param func: This function should be applicable to elements of
    the list l. E.g. a lamda func is also sufficient.
    :param message: A string that you want to be displayed
    :return: The result of map(func, l)
    """

    l_copy = l.copy()
    n_elements = len(l)
    bar = ChargingBar(message, max=n_elements)

    for idx in np.arange(n_elements):
        l_copy[idx] = func(l[idx])
        bar.next()

    bar.finish()
    return l_copy


def progress_bar_wrapper(func,
                         l,
                         message='Processing...'):
    """
    !
    :param l: List of elements
    :param func: This function should be applicable to elements of
    the list l. E.g. a lamda func is also sufficient.
    :param message: A string that you want to be displayed
    :return: The result of map(func, l)
    """

    l_copy = l.copy()
    n_elements = len(l)

    for idx in tqdm(np.arange(n_elements), desc=message):
        l_copy[idx] = func(l[idx])

    return l_copy


def test():
    M = int(10e7)
    size = int(10e4)
    l = np.random.uniform(low=-M, high=M, size=size)
    funcs = {
        'const_mul': lambda x: x*2,
        'power_2': lambda x: x**2,
        'subtraction': lambda x: x-x/2.
    }

    for name, func in funcs.items():
        map_result = list(map(func, l))
        wrapper_result = progress_bar_wrapper(func, l, message=name)
        assert any(map_result ==wrapper_result), 'Progress wrapper ' \
                                                 'should provide the ' \
                                                 'same result as map'

if __name__ == "__main__":
    test()