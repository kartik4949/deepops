import unittest
import functools
import logging
import timeit

import numpy as np
import pytest

import deepops as dp


def _helper_test(shps, lmts, np_fxn, deepops_fxn, atol=1e-6, rtol=1e-6):
    # random tensor
    np.random.seed(111)
    numpy_array = [
        np.asarray(np.random.uniform(l[0], l[-1], size=s), dtype=np.float32)
        for s, l in zip(shps, lmts)
    ]
    dp_tensors = [dp.Tensor(na).device("gpu") for na in numpy_array]

    # log speeds
    deepop_fp = (
        timeit.Timer(functools.partial(deepops_fxn, *dp_tensors)).timeit(5) * 1000 / 5
    )
    np_fp = timeit.Timer(functools.partial(np_fxn, *numpy_array)).timeit(5) * 1000 / 5
    out = deepops_fxn(*dp_tensors)
    np_out = np_fxn(*numpy_array)

    np.testing.assert_allclose(out, np_out, atol=atol, rtol=rtol)

    print(
        "\nTesting the speed || DeepOps %s function, took %.2f ms and Numpy took %.2f ms"
        % (deepops_fxn.__name__, deepop_fp, np_fp),
    )


class TestDeviceTransfers(unittest.TestCase):
    def test_host_device(self):
        a = np.asarray(np.random.uniform(0, 1, size=[1, 10000]), dtype=np.float32)
        tensor = dp.Tensor(a)
        tensor.device("gpu")
        # TODO (we can do it better here.)
        self.assertIsNotNone(tensor.device_data)

    @pytest.mark.skip
    def test_multidevice_op(self):
        a = np.asarray(np.random.uniform(0, 1, size=[1, 10000]), dtype=np.float32)
        b = np.asarray(np.random.uniform(0, 1, size=[1, 10000]), dtype=np.float32)
        tensora = dp.Tensor(a)
        tensorb = dp.Tensor(b)
        tensora.device("gpu")
        tensorb.device("gpu:1")
        # TODO (test different gpu stuff)


class TestOps(unittest.TestCase):
    def testAdd(self):
        _helper_test(
            [(1, 1000000), (1, 1000000)],
            [(-1, 1), (-1, 1)],
            np.add,
            dp.Tensor.add,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
