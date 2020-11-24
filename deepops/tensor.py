from enum import Enum
import numpy as np

class TensorState(Enum):
    """Tensor State which tells the current state of Tensor i.e
       Detach, Device, Host.
    """
    DEVICE = 'DEVICE'
    HOST = 'HOST'
    DETACH = 'DETACH'

class Tensor(self):
    """Tensor Class"""
    def __init__(self, data, gpu=None):
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
        if isinstance(data,list):
            data = np.asarray(data)
        assert isinstance(data,np.ndarray), f"numpy excepted but {type(data)} passed."
        self.data = data
        self.gpu = gpu
        self.state = TensorState.HOST
        self.device_name = 'cpu:0'

    def detach(self):
        self.state = TensorState.DETACH
        return Tensor(self.data, gpu=False)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def device(self, name=None):
        self.device_name = name
        # TODO(kartik4949): send data to GPU 


