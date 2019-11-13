import numpy as np
import tensorflow as tf

from elasticdl.proto import elasticdl_pb2
from elasticdl.python.common.dtypes import (
    dtype_numpy_to_tensor,
    dtype_tensor_to_numpy,
)


class Tensor(object):
    """Data structure for tensors in ElasticDL.

    `Tensor` can save dense tensors and sparse tensors. For sparse tensors,
    this structure saves them in the same way as `TensorFlow.IndexedSlices`.
    """

    def __init__(self, values=None, indices=None, name=None, dense_shape=None):
        """
        `Tensor` can save dense tensors and sparse tensors.
        To pass in a dense tensor, `values` should be `numpy.ndarray`,
            `indices` and `dense_shape` should be None.
        There are two ways to pass in a sparse tensor:
            * `values` is a `numpy.ndarray`, `indices` and `dense_shape` are
                `numpy.ndarray`s.
            * `values` is a `TensorFlow.IndexedSlices`, `indices` and
                `dense_shape` are None.

        Args:
            values: A `numpy.ndarray` or `TensorFlow.IndexedSlices`.
                If `values` is a `TensorFlow.IndexedSlices`, `indices` should
                be None.
            indices: A `numpy.ndarray` or None.
            name: A python string.
            dense_shape: The shape of the original tensor if `Tensor` saves a
                sparse tensor.
        """
        self.set(values, indices, name, dense_shape)

    @classmethod
    def from_tensor_pb(cls, tensor_pb):
        """Create an ElasticDL Tensor object from tensor protocol buffer.

        Return the created Tensor object.
        """
        tensor = cls()
        deserialize_tensor_pb(tensor_pb, tensor)
        return tensor

    def set(self, values=None, indices=None, name=None, dense_shape=None):
        self.name = name
        if isinstance(values, tf.IndexedSlices):
            if indices is not None or dense_shape is not None:
                raise ValueError(
                    "When creating a Tensor object with values of type "
                    "tf.IndexedSlices, indices and dense shape must be None."
                )

            self.values = values.values.numpy()
            self.indices = values.indices.numpy()
            self.dense_shape = (
                tuple(values.dense_shape)
                if values.dense_shape is not None
                else None
            )
        else:
            if dense_shape is not None and values.shape[1] != dense_shape[1]:
                raise ValueError(
                    "Values and dense_shape have incompatible shape, %s and "
                    "%s." % (str(values.shape), str(dense_shape))
                )
            self.values = (
                values.numpy() if isinstance(values, tf.Tensor) else values
            )
            self.indices = (
                indices.numpy() if isinstance(indices, tf.Tensor) else indices
            )
            self.dense_shape = (
                tuple(dense_shape) if dense_shape is not None else None
            )

    def is_indexed_slices(self):
        return self.indices is not None

    def to_tensor_pb(self):
        tensor_pb = elasticdl_pb2.Tensor()
        serialize_tensor(self, tensor_pb)
        return tensor_pb

    def to_tf_tensor(self):
        if self.is_indexed_slices():
            return tf.IndexedSlices(
                self.values, self.indices, self.dense_shape
            )
        else:
            return tf.constant(self.values)

    def to_ndarray(self):
        if self.is_indexed_slices():
            # Currently Tensor does not have a field representing dense shape,
            # thus can not convert it to numpy.ndarray.
            if self.dense_shape is None:
                raise ValueError(
                    "Attempted to convert IndexedSlices without dense shape "
                    "to numpy ndarray."
                )
            return tf.math.unsorted_segment_sum(
                self.values, self.indices, self.dense_shape[0]
            ).numpy()
        return self.values

    def __add__(self, other):
        if self.is_indexed_slices() and other.is_indexed_slices():
            self.values = np.concatenate((self.values, other.values), axis=0)
            self.indices = np.concatenate(
                (self.indices, other.indices), axis=0
            )
        elif not self.is_indexed_slices() and not other.is_indexed_slices():
            self.values = self.values + other.values
        else:
            raise NotImplementedError(
                "Only Tensor with the same type could be added"
            )
        return self

    def __radd__(self, other):
        return self + other


def serialize_tensor(tensor, tensor_pb):
    """Serialize ElasticDL Tensor to tensor protocol buffer."""
    dtype = dtype_numpy_to_tensor(tensor.values.dtype)
    if not dtype:
        raise ValueError(
            "Dtype of ndarray %s is not supported", tensor.values.dtype
        )
    tensor_pb.dtype = dtype
    tensor_pb.dim.extend(tensor.values.shape)
    tensor_pb.content = tensor.values.tobytes()
    if tensor.is_indexed_slices():
        tensor_pb.indices.extend(tuple(tensor.indices))
    if tensor.name:
        tensor_pb.name = tensor.name


def deserialize_tensor_pb(tensor_pb, tensor):
    """Deserialize tensor protocol buffer to ElasticDL Tensor.

    Note that the input tensor protocol buffer is reset and underlying buffer
    is passed to the returned ndarray.
    """
    if not tensor_pb.dim:
        raise ValueError("Tensor PB has no dim defined")

    dtype = dtype_tensor_to_numpy(tensor_pb.dtype)
    # Check that the buffer size agrees with dimensions.
    size = dtype.itemsize
    for d in tensor_pb.dim:
        size *= d
    if size != len(tensor_pb.content):
        raise ValueError(
            "Tensor PB size mismatch, dim: %s, len(content): %d",
            tensor_pb.dim,
            len(tensor_pb.content),
        )
    tensor.set(
        values=np.ndarray(
            shape=tensor_pb.dim, dtype=dtype, buffer=tensor_pb.content
        ),
        indices=np.array(tensor_pb.indices) if tensor_pb.indices else None,
        name=tensor_pb.name,
    )
    tensor_pb.Clear()


def tensor_pb_to_ndarray(tensor_pb):
    """Deserialize tensor protocol buffer and return a numpy ndarray."""
    return Tensor.from_tensor_pb(tensor_pb).to_ndarray()


def tensor_pb_to_tf_tensor(tensor_pb):
    """Deserialize tensor protocol buffer and return a TensorFlow tensor."""
    return Tensor.from_tensor_pb(tensor_pb).to_tf_tensor()


def emplace_tensor_pb_from_ndarray(
    tensor_pb_list, values, indices=None, name=None
):
    """Generate a tensor procotol buffer and append it to tensor_pb_list.

    Note:
        This function does not use list append function as following code
            snippet. It is slow because append function will copy the input
            protocol buffer.

        ```
        pb = elasticdl_pb2.Tensor()
        pb.dim.extend([3])
        pb.name = "test"
        pb.dtype = DT_INT64
        pb.content = np.array([1, 2, 3]).tobytes()
        tensor_pb_list.append(tensor_pb) # slow, because append copies pb
        ```
    """
    tensor_pb = tensor_pb_list.add()
    tensor = Tensor(values, indices, name)
    serialize_tensor(tensor, tensor_pb)
