import numpy as np

def normalize(v):
	n = np.linalg.norm(v)
	return v/n if not np.isclose(n, 0) else v