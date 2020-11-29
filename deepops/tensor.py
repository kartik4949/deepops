"""Tensor Class."""
import functools
import operator

import numpy as np

# PyCUDA initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .gpu_kernels import add, arithmetic
from .states import TensorState


ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


class GPUConnectMixin:
    """Mixin for GPU connect"""

    def _alloc_device_memory(self, shape):
        """_alloc_device_memory.
        Allocate memory  to device.

        Args:
            data:
        """
        _nbytes = np.prod(shape) * 4
        _device_data = cuda.mem_alloc(int(_nbytes))
        _device_data.shape = tuple(shape)
        _device_data.dtype = np.float32
        return _device_data

    def _memory_host_to_device(self, device_data, data):
        """_memory_host_to_device.
        Copy memory host to device(GPU).

        Args:
            data:
            device_data:
        """
        cuda.memcpy_htod(device_data, data)
        return

    @staticmethod
    def _idiv(a, b):
        return a // b + 1

    @staticmethod
    def get_kernel(kernel, function):
        """get_kernel.
        get the kernel.

        Args:
            kernel:
            function:
        """
        return kernel.get_function(function)


class GradientMixin:
    def _walk(self, leaf_out_node, in_grad):
        """_walk.
        Reverse Graph Traversal with gradients.

        Args:
            leaf_out_node: Leaf Node.
            in_grad: Input Gradient.
        """
        _child_nodes = leaf_out_node._child_nodes
        _in_grads = leaf_out_node._backward(in_grad)
        _ = [
            self._walk(node, in_grad) for node, in_grad in zip(_child_nodes, _in_grads)
        ]
        return in_grad

    def backward(self):
        """backward.
        Backward Function with Input Gradient set 1.

        Args:
            out_node: Leaf Output Node.
        """
        gradient = self._walk(self, 1)
        self.grad = gradient
        return gradient


class Tensor(GPUConnectMixin, GradientMixin):
    """Tensor Class."""

    BLOCKSIZE = 256

    """
    The dict wastes a lot of RAM. Python can’t just allocate a static amount of memory at
    object creation to store all the attributes. Therefore it sucks a lot of RAM if you
    create a lot of objects (I am talking in thousands and millions).
    Still there is a way to circumvent this issue.
    It involves the usage of __slots__ to tell Python not to use a dict,
    and only allocate space for a fixed set of attributes.
    """
    __slots__ = ("_data", "_name", "_dtype", "_shape", "gpu", "state", "device_name")

    def __init__(self, data, name=None, dtype=None):
        """__init__.
        Initializes Tensor Class.

        Args:
            data: list or np.array data.
            gpu: use gpu?
        ::

        Example:

        >> a = Tensor([1, 2])
        >> b = Tensor([2,3])
        >> print(a + b)
        (dp.Tensor, shape=(2,), dtype = int32, numpy:([3,5], dtype = int32)
        """
        self.state = TensorState.HOST
        if isinstance(data, list) or isinstance(data, float):
            data = np.array(data, dtype=dtype if dtype else np.float32)
        elif isinstance(data, pycuda._driver.DeviceAllocation):
            self.state = TensorState.DEVICE
        elif not (isinstance(data, np.ndarray) or isinstance(data, np.float32)):
            raise TypeError(f"numpy excepted but {type(data)} passed.")
        self._data = data
        self._dtype = data.dtype
        self._shape = data.shape
        self._name = name
        self.gpu = False
        self.grad = 0.0
        self._child_nodes = tuple()

        def _backward(in_grad=1):
            self.grad = in_grad
            return (in_grad,)

        self._backward = _backward
        self.device_name = "cpu:0"

    def detach(self):
        """detach.
        Detach state.
        """
        self.state = TensorState.DETACH
        # TODO(kartik4949) : Write ME.
        return Tensor(self.data)

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def where(self):
        return self._device()

    def _device(self):
        if self.state == TensorState.DEVICE:
            _cuda_device = "gpu"

        if self.state == TensorState.HOST:
            _cuda_device = "cpu"
        return _cuda_device

    def asarray(self, data: list = None, dtype: tuple = None):
        """asarray.
        convert array to DP array.

        Args:
            data (list): data
            dtype (tuple): dtype
        """
        # Depracted!
        return Tensor(np.asarray(data, dtype=dtype))

    def device(self, name: str = None):
        """device.
        register the data on device.

        Args:
            name (str): name of device
        """
        assert name.startswith("cpu") or name.startswith("gpu"), "Wrong Device!!"
        # set precision to float32.
        assert (
            self.dtype == np.float32
        ), "Only single precision is supported i.e float32"
        if self.state != TensorState.DEVICE:
            self.state = TensorState.DEVICE
            self.device_name = name
            data = self._alloc_device_memory(self.shape)
            self._memory_host_to_device(data, self.data)
            self._shape = self.data.shape
            self._dtype = self.data.dtype
            self._data = data
        return self

    def cpu(
        self,
    ):
        """cpu.
        copy buffer from device to cpu.
        """
        _host_out_arry = np.empty(self.shape, dtype=np.float32)
        cuda.memcpy_dtoh(_host_out_arry, self._data)
        cuda.Context.synchronize()
        return Tensor(_host_out_arry)

    def add(self, tensor):
        """add.
        Vector Addition which adds Tensor with given Tensor.

        Args:
            tensor: Tensor class
        """

        def _backward(in_grad):
            self_grad = in_grad
            tensor_grad = in_grad
            self.grad = self_grad
            tensor.grad = tensor.grad
            return self_grad, tensor_grad

        return self.arithmetic(tensor, _backward, "+")

    def sub(self, tensor):
        """sub.
        Vector Addition which substracts Tensor with given Tensor.

        Args:
            tensor: Tensor class
        """

        def _backward(in_grad):
            self_grad = in_grad
            tensor_grad = -in_grad
            self.grad = self_grad
            tensor.grad = tensor.grad
            return self_grad, tensor_grad

        return self.arithmetic(tensor, _backward, "-")

    def mul(self, tensor):
        """mul.
        Vector Addition which multiplies Tensor with given Tensor.

        Args:
            tensor: Tensor class
        """

        def _backward(in_grad):
            self_grad = in_grad * tensor.data
            tensor_grad = in_grad * self.data
            self.grad = self_grad
            tensor.grad = tensor_grad
            return self_grad, tensor_grad

        return self.arithmetic(tensor, _backward, "*")

    def arithmetic(self, tensor, backward=None, operation: str = "+"):
        """Arithmetic.
        Vector arithmetic operations on given Tensor.

        Args:
            tensor: Tensor class
        """
        if self.state != TensorState.DEVICE:
            ret = Tensor(ops[operation](self.data, tensor.data))
            ret._child_nodes = (self, tensor)
            if backward:
                ret._backward = backward
            return ret
        assert isinstance(
            tensor, self.__class__
        ), f"Tensor is required but passed {type(tensor)}"
        ret = self._alloc_device_memory(self.shape)
        N = max(self.shape)
        blockDim = (self.BLOCKSIZE, 1, 1)
        gridDim = (self._idiv(N, self.BLOCKSIZE), 1, 1)
        _vec_kernel = self.get_kernel(arithmetic(operation), "device_arithmetic")
        _vec_kernel(
            ret,
            self.data,
            tensor.data,
            np.int32(N),
            block=blockDim,
            grid=gridDim,
        )

        ret = Tensor(ret)
        ret._dtype = self.dtype
        ret._shape = self.shape
        return ret

    def __add__(self, tensor):
        return self.add(tensor)

    def __mul__(self, tensor):
        return self.mul(tensor)

    def __substract__(self, tensor):
        return self.sub(tensor)

    def __repr__(self):
        return "dp.Tensor %s shape: %s, numpy: (%s, dtype=%s), cuda device: %s" % (
            f"name: {self.name}, " if self.name else "",
            self.shape,
            self.data,
            self.dtype,
            self.where,
        )
