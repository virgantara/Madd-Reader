import numpy as np
def format_vector(features: np.array):
  x = []
  for i in range(len(features)):
    if len(x) == 0:
      x = features[i]
    else:
      x = np.append(x, features[i], axis=0)

  return x