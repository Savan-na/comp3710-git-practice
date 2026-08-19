import torch
import numpy as np

print("PyTorch Version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

# Y,X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
# Y, X = np.mgrid[-0.9:-0.5:0.001, -0.7:-0.3:0.001]
# Y, X = np.mgrid[-0.7:-0.6:0.00025, -0.65:-0.50:0.00025]

# What: Zoom further into the star-shaped boundary region selected from the previous image.
# Why: A smaller window and finer spacing reveal finer-scale fractal structure around this boundary.
# Y, X = np.mgrid[   -0.67750:-0.67250:0.00001,   -0.61375:-0.60875:0.00001]

Y, X = np.mgrid[
    -0.68000:-0.67575:0.00001,
    -0.61575:-0.61125:0.00001
]



print("X shape:", X.shape)
print("Y shape:", Y.shape)

x = torch.Tensor(X)
y = torch.Tensor(Y)

z = torch.complex(x,y)

print("z shape:", z.shape)
print("z dtype:", z.dtype)
print("z device before:", z.device)

zs = z.clone()

ns = torch.zeros_like(z, dtype=torch.float32)

print("zs shape:", zs.shape)
print("ns shape:", ns.shape)

z = z.to(device)
zs = zs.to(device)

ns = ns.to(device)

print("z device after:",z.device)
print("zs device after:", zs.device)
print("ns device after:", ns.device)

for i in range(200):
	zs_ = zs * zs + z
	not_diverged = torch.abs(zs_) < 4.0
	ns += not_diverged
	zs = zs_

print("ns max:", ns.max().item())
print("ns min:", ns.min().item())

import matplotlib.pyplot as plt

def processFractal(a):
	a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
	img = np.concatenate([
		10 + 20*np.cos(a_cyclic),
		30 + 50*np.sin(a_cyclic),
		155 - 80*np.cos(a_cyclic)
	],2)

	img[a == a.max()] = 0
	img = np.uint8(np.clip(img, 0,255))

	return img


ns_plot = ns.cpu().numpy()

plt.figure(figsize=(16,10))
plt.imshow(processFractal(ns_plot))
plt.tight_layout(pad=0)
plt.savefig("mandelbrot_zoom.png")
plt.show()
