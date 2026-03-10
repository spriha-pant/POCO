import numpy as np
d = np.load("barik_mouse_mousmi1.npz")
print(d.files)
print(d["M"].shape)