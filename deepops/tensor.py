"""Tensor Class."""
import numpy as np

# PyCUDA initialization
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from .gpu_kernels import add
from .states import TensorState


class GPUConnectMixin:
    """Mixin for GPU connect"""

    def _alloc_devic_memory(self, data):
        """_alloc_devic_memory.
        Allocate memory  to device.

        Args:
            data:
        """
        _device_data = cuda.mem_alloc(data.nbytes)
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


class Tensor(GPUConnectMixin):
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
    __slots__ = ("data", "gpu", "state", "device_name")

    def __init__(self, data):
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
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        assert isinstance(data, np.ndarray), f"numpy excepted but {type(data)} passed."

        # set precision to float32.
        assert (
            data.dtype == np.float32
        ), "Only single precision is supported i.e float32"
        self.data = data
        self.gpu = False
        self.state = TensorState.HOST
        self.device_name = "cpu:0"

    def detach(self):
        """detach.
        Detach state.
        """
        self.state = TensorState.DETACH
        # TODO(kartik4949) : Write ME.
        return Tensor(self.data, gpu=False)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def device(self, name: str = None):
        """device.
        register the data on device.

        Args:
            name (str): name of device
        """
        assert name.startswith("cpu") or name.startswith("gpu"), "Wrong Device!!"
        self.state = TensorState.DEVICE
        self.device_name = name
        self.device_data = self._alloc_devic_memory(self.data)
        self._memory_host_to_device(self.device_data, self.data)
        return self

    def add(self, tensor):
        """add.
        Vector Addition which adds Tensor with given Tensor.

        Args:
            tensor: Tensor class
        """
        assert isinstance(
            tensor, self.__class__
        ), f"Tensor is required but passed {type(tensor)}"
        out_data = self._alloc_devic_memory(self.data)
        N = max(self.data.shape)
        blockDim = (self.BLOCKSIZE, 1, 1)
        gridDim = (self._idiv(N, self.BLOCKSIZE), 1, 1)
        _vec_add_kernel = self.get_kernel(add, "device_vec_add")
        _vec_add_kernel(
            out_data,
            self.device_data,
            tensor.device_data,
            np.int32(N),
            block=blockDim,
            grid=gridDim,
        )
        _host_out_arry = np.empty_like(self.data)
        cuda.memcpy_dtoh(_host_out_arry, out_data)
        cuda.Context.synchronize()
        return _host_out_arry

    def __add__(self, tensor):
        return self.add(tensor)

    def __repr__(self):
        return f"dp.Tensor shape={self.shape}, numpy=({self.data}, dtype={self.data.dtype})"
