import tensorflow as tf


# def stack_nested_tensors(tensors):
#   """Stacks a list of nested tensors along the first dimension.
#   Args:
#     tensors: A list of nested tensors to be stacked along the first dimension.
#   Returns:
#     A stacked nested tensor.
#   """
#   return tf.nest.map_structure(lambda *tensors: tf.stack(tensors), *tensors)
#
#
# def dict_gather(data, k):
#     output = dict()
#
#     for key, value in data.items():
#         output[key] = value[:, k]
#
#     return output
