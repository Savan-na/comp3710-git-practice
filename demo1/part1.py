import torch

import numpy as np

# print("PyTorch Version:", torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:",device)

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# print("X shape:", X.shape)
# print("Y shape:", Y.shape)

x = torch.Tensor(X)
y = torch.Tensor(Y)

# print("x shape:", x.shape)
# print("y shape:", y.shape)
# print("x device before:", x.device)
# print("y device before:", y.device)

x = x.to(device)
y = y.to(device)

# print("x.device after:", x.device)
# print("y device after:", y.device)

z = torch.exp(-(x**2 + y**2) / 2.0)

print("z shape:", z.shape)
print("z device:",z.device)
print("z max:", z.max().item())
print("z min:",z.min().item())

import matplotlib.pyplot as plt

z_plot = z.cpu().numpy()

plt.imshow(z_plot)
plt.tight_layout()

plt.savefig("Gaussian.png")

plt.show()

sine = torch.sin(x + y)

print("sine shape:", sine.shape)
print("sine device:",sine.device)
print("sine max:", sine.max().item())
print("sine min:", sine.min().item())

sine_plot = sine.cpu().numpy()

plt.figure()
plt.imshow(sine_plot)

plt.tight_layout()
plt.savefig("sine.png")
plt.show()

gabor = z * sine

print("gabor shape:", gabor.shape)
print("gabor device:", gabor.device)
print("gabor max:", gabor.max().item())
print("gabor min:", gabor.min().item())

gabor_plot = gabor.cpu().numpy()

plt.figure()
plt.imshow(gabor_plot)
plt.tight_layout()
plt.savefig("gabor.png")
plt.show()

