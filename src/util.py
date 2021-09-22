import numpy as np

data_field = lambda it: it.data

concat_fields = lambda elements, get_field_fn: np.concatenate(list(map(get_field_fn, elements)))