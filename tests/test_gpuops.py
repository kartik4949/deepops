import unittest
import re
import functools
import logging
import timeit

import numpy as np
import tensorflow as tf
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
    out = deepops_fxn(*dp_tensors).cpu().data
    np_out = np_fxn(*numpy_array)

    np.testing.assert_allclose(out, np_out, atol=atol, rtol=rtol)

    logging.info(
        "\nTesting the speed || DeepOps %s function, took %.2f ms and Numpy took %.2f ms",
        (deepops_fxn.__name__, deepop_fp, np_fp),
    )


def _helper_test_backward(shps, lmts, tf_fxn, deepops_fxn, atol=1e-6, rtol=1e-6):
    # random tensor
    tf.random.set_seed(111)
    tf_arrays = [
        tf.Variable(tf.random.uniform(s, l[0], l[-1], dtype=tf.float32))
        for s, l in zip(shps, lmts)
    ]
    dp_tensors = [dp.Tensor(na.numpy()) for na in tf_arrays]

    out = deepops_fxn(*dp_tensors)
    out.backward()
    with tf.GradientTape() as gt:
        out_tf = tf_fxn(*tf_arrays)
    out_tf = gt.gradient(out_tf, tf_arrays)
    for dp_tensor, tf_grad in zip(dp_tensors, out_tf):
        np.testing.assert_allclose(
            dp_tensor.grad, tf_grad.numpy(), atol=atol, rtol=rtol
        )


class TestDeviceTransfers(unittest.TestCase):
    def test_host_device(self):
        a = np.asarray(np.random.uniform(0, 1, size=[1, 10000]), dtype=np.float32)
        tensor = dp.Tensor(a)
        tensor.device("gpu")
        # TODO (we can do it better here.)
        self.assertIsNotNone(tensor.data)

    @pytest.mark.skip
    def test_multidevice_op(self):
        a = np.asarray(np.random.uniform(0, 1, size=[1, 10000]), dtype=np.float32)
        b = np.asarray(np.random.uniform(0, 1, size=[1, 10000]), dtype=np.float32)
        tensora = dp.Tensor(a)
        tensorb = dp.Tensor(b)
        tensora.device("gpu")
        tensorb.device("gpu:1")
        # TODO (test different gpu stuff)


class TestOpsForward(unittest.TestCase):
    def test_add(self):
        _helper_test(
            [(1, 1000000), (1, 1000000)],
            [(-1, 1), (-1, 1)],
            np.add,
            dp.Tensor.add,
        )

    def test_mul(self):
        np_mul = lambda a, b: a * b
        _helper_test(
            [(1, 1000000), (1, 1000000)],
            [(-1, 1), (-1, 1)],
            np_mul,
            dp.Tensor.mul,
        )


class TestOpsBackward(unittest.TestCase):
    def test_add_backward(self):
        _helper_test_backward(
            [(1, 1000000), (1, 1000000)],
            [(-1, 1), (-1, 1)],
            tf.add,
            dp.Tensor.add,
        )

    def test_mul_backward(self):
        _helper_test_backward(
            [(1, 1000000), (1, 1000000)],
            [(-1, 1), (-1, 1)],
            tf.multiply,
            dp.Tensor.mul,
        )


class TestTensorPrint(unittest.TestCase):
    def test_repr(self):
        a = np.asarray(np.random.uniform(0, 1, size=[1, 10000]), dtype=np.float32)
        dp_a = dp.Tensor(a)
        repr_a = dp_a.__repr__()
        shape = re.search("shape: \((.*?)\)", repr_a).groups()[0]
        self.assertEqual("(" + shape + ")", str(a.shape))


if __name__ == "__main__":
    unittest.main(verbosity=2)
