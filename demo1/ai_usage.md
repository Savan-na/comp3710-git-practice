# Task 1

## Question
Generate a Python script to plot a 2D Gaussian function using NumPy and Matplotlib. Please answer me in English.

## Answer
This directly matches the **Part 1.2 AI Task** in your COMP3710 Demo 1 sheet: generate the 2D Gaussian using **NumPy + Matplotlib** rather than PyTorch. 

```python
import numpy as np
import matplotlib.pyplot as plt

# What: Create a 2D coordinate grid from -4 to 4.
# Why: The Gaussian value needs to be calculated at every (x, y) position.
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

# What: Compute the 2D Gaussian function.
# Why: Points near (0, 0) have larger values, while distant points decay toward 0.
Z = np.exp(-(X**2 + Y**2) / 2.0)

# What: Display the Gaussian as an image.
# Why: imshow converts the 2D array of Gaussian values into a visible intensity pattern.
plt.imshow(Z)
plt.tight_layout()
plt.show()
```

The key mathematical line is:

[
z=e^{-\frac{x^2+y^2}{2}}
]

So the correspondence is very direct:

```text
Mathematics                     NumPy
------------------------------------------------
x, y coordinates        →       X, Y
x² + y²                 →       X**2 + Y**2
-(x² + y²) / 2          →       -(X**2 + Y**2) / 2.0
e^(...)                 →       np.exp(...)
Gaussian values         →       Z
```

Unlike the PyTorch version in the lab, this NumPy version does **not** need `torch.Tensor()`, `.to(device)`, or `.cpu().numpy()`; 
`Z` is already a NumPy array that Matplotlib can plot directly. The course materials specifically expect familiarity with NumPy and Matplotlib. 

**Quick self-check:** `X.shape`, `Y.shape`, and `Z.shape` should all be `(800, 800)`. `Z.max()` should be `1.0`, because the Gaussian reaches its maximum at ((0,0)).

## Question

Convert this script to PyTorch using tensors and CUDA when available.


## Answer
This is the direct PyTorch conversion of the NumPy version, using **tensors** and automatically using **CUDA when available**,
 which is exactly the conversion requested in COMP3710 Demo 1 Part 1.2. 

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# What: Select the GPU if CUDA is available; otherwise use the CPU.
# Why: This allows the same code to run on either GPU or CPU automatically.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# What: Create a 2D coordinate grid from -4 to 4.
# Why: We need an (x, y) coordinate for every point in the Gaussian image.
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

# What: Convert the NumPy arrays into PyTorch tensors.
# Why: PyTorch operations work on tensors rather than NumPy arrays.
x = torch.Tensor(X)
y = torch.Tensor(Y)

# What: Move the tensors to the selected device.
# Why: If CUDA is available, the Gaussian calculation can run on the GPU.
x = x.to(device)
y = y.to(device)

# What: Compute the 2D Gaussian using PyTorch tensor operations.
# Why: Every (x, y) position is evaluated using the Gaussian equation.
z = torch.exp(-(x**2 + y**2) / 2.0)

# What: Move the result back to the CPU and convert it to NumPy.
# Why: Matplotlib plots NumPy arrays rather than CUDA tensors.
z_plot = z.cpu().numpy()

# What: Display the Gaussian.
# Why: This visualises the computed 2D Gaussian values.
plt.imshow(z_plot)
plt.tight_layout()
plt.show()
```

The most important conversion is:

```text
NumPy version                  PyTorch version

np.exp(...)              →     torch.exp(...)
X, Y                     →     x, y tensors
                         →     x.to(device)
                         →     y.to(device)
Z                        →     z
Z                        →     z.cpu().numpy()
                               before plotting
```

The mathematical calculation itself has **not changed**:

[
z=e^{-\frac{x^2+y^2}{2}}
]

Only the representation and execution environment changed:

```text
NumPy arrays
    ↓
PyTorch tensors
    ↓
move to CUDA if available
    ↓
torch.exp(...) on tensors
    ↓
move result back to CPU
    ↓
convert to NumPy
    ↓
Matplotlib
```

This closely follows the sequence used in the official lab: create the grid with `np.mgrid`, convert it into PyTorch tensors, transfer them with `.to(device)`, calculate the Gaussian with `torch.exp`, then use `.cpu().numpy()` for plotting. 

**Quick self-check**

```python
print(x.shape)       # expected: torch.Size([800, 800])
print(z.shape)       # expected: torch.Size([800, 800])
print(z.max())       # expected: approximately tensor(1.)
print(z.device)      # expected: cuda:0 if CUDA is available
```

If CUDA is available, the important expected result is:

```text
Device: cuda
```

and `z.device` should show something like `cuda:0`.

## Question

Modify it to create a 2D sine function whose angle depends on both x and y coordinates.

## Answer

Yes. For the COMP3710 Part 1.2 task, the sine angle should depend on **both `x` and `y` coordinates**, so a simple choice is

[
z=\sin(x+y)
]

This produces diagonal “stripes,” matching the type of output requested in the lab. 

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# What: Select CUDA if a GPU is available; otherwise use the CPU.
# Why: This lets PyTorch perform the calculation on the available device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# What: Create a 2D coordinate grid.
# Why: Each pixel needs both an x-coordinate and a y-coordinate.
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

# What: Convert the NumPy coordinate arrays into PyTorch tensors.
# Why: torch.sin() operates on PyTorch tensors.
x = torch.Tensor(X).to(device)
y = torch.Tensor(Y).to(device)

# What: Calculate a 2D sine function whose angle depends on x and y.
# Why: x + y makes the sine value change in both spatial directions.
z = torch.sin(x + y)

# What: Move the result back to the CPU for plotting.
# Why: Matplotlib cannot directly plot a CUDA tensor.
z_plot = z.cpu().numpy()

# What: Display the sine function.
# Why: The changing sine values appear visually as stripes.
plt.imshow(z_plot)
plt.tight_layout()
plt.show()
```

### The key change

Your Gaussian was:

```python
z = torch.exp(-(x**2 + y**2) / 2.0)
```

Now replace it with:

```python
z = torch.sin(x + y)
```

Why `x + y`?

```text
same x, different y → angle changes
different x, same y → angle changes

Therefore:
both x AND y affect the sine value.
```

For example:

```text
(x, y) = (0, 0)   → sin(0 + 0) = 0
(x, y) = (1, 0)   → sin(1)
(x, y) = (0, 1)   → sin(1)
(x, y) = (1, 1)   → sin(2)
```

So unlike a function such as `torch.sin(x)`, which only varies along one coordinate, `torch.sin(x + y)` genuinely depends on **both coordinates**.

**Quick self-check**

```python
print(z.shape)       # torch.Size([800, 800])
print(z.min())       # approximately -1
print(z.max())       # approximately 1
print(z.device)      # cuda:0 if CUDA is available
```

You should see repeating **diagonal light/dark stripes** in the plot.

## Question

Generate a Mandelbrot set implementation using PyTorch tensors and CUDA when available.

## Answer

Below is a PyTorch implementation of the **Mandelbrot set** using tensors and **CUDA automatically when available**. It follows the structure of COMP3710 Demo 1 Part 2: create a complex-plane grid, iterate (z_{n+1}=z_n^2+c), count how long each point remains bounded, and visualise the result. 

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# What: Select CUDA if available; otherwise use the CPU.
# Why: The Mandelbrot calculation can evaluate many points in parallel on a GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# What: Create coordinates covering part of the complex plane.
# Why: Every position represents one possible value of c = x + yi.
Y, X = np.mgrid[-1.3:1.3:0.005, -2.0:1.0:0.005]

# What: Convert the coordinate arrays into PyTorch tensors.
# Why: PyTorch can perform the Mandelbrot calculations in parallel.
x = torch.Tensor(X).to(device)
y = torch.Tensor(Y).to(device)

# What: Combine x and y into complex numbers c = x + yi.
# Why: The Mandelbrot set is defined in the complex plane.
c = torch.complex(x, y)

# What: Start z at zero for every point.
# Why: Mandelbrot iteration always begins with z_0 = 0.
z = torch.zeros_like(c)

# What: Store how many iterations each point remains bounded.
# Why: This lets us distinguish points that diverge quickly from those that remain bounded.
iterations = torch.zeros(c.shape, dtype=torch.int32, device=device)

max_iterations = 200

# What: Repeatedly apply z_(n+1) = z_n^2 + c.
# Why: Membership in the Mandelbrot set depends on whether this sequence stays bounded.
for i in range(max_iterations):

    z = z * z + c

    # What: Find points that have not escaped.
    # Why: A magnitude above the escape threshold indicates divergence.
    not_diverged = torch.abs(z) < 4.0

    # Count one more iteration for points that are still bounded.
    iterations += not_diverged

# What: Move the result from GPU to CPU and convert it to NumPy.
# Why: Matplotlib plots CPU/NumPy data.
mandelbrot = iterations.cpu().numpy()

# What: Display the Mandelbrot set.
# Why: Iteration counts reveal the fractal structure.
plt.figure(figsize=(12, 8))
plt.imshow(mandelbrot, extent=[-2, 1, -1.3, 1.3], origin="lower")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Mandelbrot Set")
plt.tight_layout()
plt.show()
```

The central logic is only this:

```python
z = z * z + c
not_diverged = torch.abs(z) < 4.0
iterations += not_diverged
```

Mathematically:

[
z_0=0
]

[
z_{n+1}=z_n^2+c
]

and each pixel represents a different complex number

[
c=x+yi.
]

So conceptually:

```text
2D pixel position
      ↓
(x, y)
      ↓
c = x + yi
      ↓
start z = 0
      ↓
z = z² + c
      ↓
z = z² + c
      ↓
z = z² + c
      ↓
repeat up to 200 times
      ↓
How long did the point remain bounded?
      ↓
pixel colour
```

One small difference from the lab-sheet code is that I used an **integer tensor** for `iterations` rather than `torch.zeros_like(c)`. Since `c` is complex-valued but the iteration count is conceptually an integer, this keeps the two roles explicit.

### Quick self-check

Add:

```python
print("c shape:", c.shape)
print("c dtype:", c.dtype)
print("iterations shape:", iterations.shape)
print("minimum:", iterations.min())
print("maximum:", iterations.max())
print("device:", c.device)
```

With the lab's grid spacing, you should get approximately:

```text
c shape: torch.Size([520, 600])
c dtype: torch.complex64
iterations shape: torch.Size([520, 600])
maximum: tensor(200, ...)
device: cuda:0
```

if CUDA is available. This also matches the lab's requirement that changing the `mgrid` position and spacing lets you later **zoom into a high-resolution region of the Mandelbrot set**. 

# Task 2

## Question

## Prompt 1 — Generate the Mandelbrot set using PyTorch and GPU

I am completing a university lab on generating the Mandelbrot set using PyTorch.

Please write Python code that computes and displays the Mandelbrot set using the iteration

\[
z_{n+1}=z_n^2+c,\qquad z_0=0
\]

Requirements:

- Use PyTorch for the Mandelbrot computation.
- Automatically use CUDA GPU if `torch.cuda.is_available()`, otherwise use CPU.
- Use complex-valued PyTorch tensors.
- Compute all Mandelbrot iterations on the GPU when CUDA is available.
- Use approximately 200 iterations.
- Use an escape radius of 4.
- Generate a 2D grid covering approximately:
  - real axis: -2 to 1
  - imaginary axis: -1.3 to 1.3
- Store an iteration/escape count for every point so that points inside the Mandelbrot set are black and points outside can be coloured according to how quickly they diverge.
- Use Matplotlib only for displaying the final result.
- Avoid transferring tensors between CPU and GPU inside the iteration loop.
- Explain the important parts of the code, especially the tensor shapes, complex tensor creation, escape mask, and GPU usage.

Please provide a complete runnable code example.

## Answer

Below is a complete runnable PyTorch implementation. It follows the core structure used in COMP3710 Demo 1 Part 2: represent each complex-plane point as a PyTorch complex tensor, iterate (z_{n+1}=z_n^2+c), track divergence, and move the result back to the CPU only after the computation is finished. 

```python
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Select the computational device
# ---------------------------------------------------------

# What: Use CUDA when available, otherwise fall back to CPU.
# Why: The same program can run on both GPU and CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# 2. Mandelbrot parameters
# ---------------------------------------------------------

max_iterations = 200
escape_radius = 4.0

width = 600
height = 520


# ---------------------------------------------------------
# 3. Create the 2D complex-plane grid
# ---------------------------------------------------------

# What: Create coordinates directly on the selected device.
# Why: If CUDA is available, the grid begins on the GPU and
#      does not need to be transferred later.
real = torch.linspace(
    -2.0, 1.0,
    width,
    dtype=torch.float32,
    device=device
)

imag = torch.linspace(
    -1.3, 1.3,
    height,
    dtype=torch.float32,
    device=device
)

# What: Generate every (x, y) coordinate pair.
# Why: Each coordinate corresponds to one pixel / one value of c.
Y, X = torch.meshgrid(imag, real, indexing="ij")

# X.shape = [520, 600]
# Y.shape = [520, 600]


# ---------------------------------------------------------
# 4. Construct complex-valued PyTorch tensors
# ---------------------------------------------------------

# What: Construct c = x + yi.
# Why: The Mandelbrot set is defined over the complex plane.
c = torch.complex(X, Y)

# What: Start every Mandelbrot sequence at z_0 = 0.
# Why: This is part of the definition of the Mandelbrot set.
z = torch.zeros_like(c)

# c and z are complex64 tensors with shape [520, 600].


# ---------------------------------------------------------
# 5. Store escape information
# ---------------------------------------------------------

# What: Initially assume every point survives all 200 iterations.
# Why: Points that never escape will remain equal to max_iterations.
escape_count = torch.full(
    c.shape,
    max_iterations,
    dtype=torch.int32,
    device=device
)

# What: Keep track of points that have not escaped yet.
# Why: We only want to record the FIRST iteration at which
#      each point escapes.
active = torch.ones(
    c.shape,
    dtype=torch.bool,
    device=device
)


# ---------------------------------------------------------
# 6. Mandelbrot iteration
# ---------------------------------------------------------

for i in range(max_iterations):

    # What: Apply z_(n+1) = z_n^2 + c simultaneously
    #       to every point in the grid.
    # Why: PyTorch performs these tensor operations in parallel.
    z = z * z + c

    # What: Detect points whose magnitude reaches the escape radius.
    # Why: These points are considered to have diverged.
    escaped_now = active & (torch.abs(z) >= escape_radius)

    # What: Record when each newly escaped point diverged.
    # Why: This value can later determine its colour.
    escape_count[escaped_now] = i + 1

    # What: Remove escaped points from the active mask.
    # Why: Their first escape iteration has already been recorded.
    active = active & (~escaped_now)


# ---------------------------------------------------------
# 7. Move the final result back to the CPU
# ---------------------------------------------------------

# What: Transfer only the completed result to the CPU.
# Why: No CPU/GPU transfers occur inside the iteration loop.
mandelbrot = escape_count.cpu().numpy()


# ---------------------------------------------------------
# 8. Prepare the image
# ---------------------------------------------------------

# Points that survived all 200 iterations are treated as
# belonging to the Mandelbrot set.
#
# Give those points a value below vmin so Matplotlib can
# display them in black.
display_image = mandelbrot.astype(float)
display_image[mandelbrot == max_iterations] = -1

cmap = plt.get_cmap("turbo").copy()
cmap.set_under("black")


# ---------------------------------------------------------
# 9. Display the Mandelbrot set
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

plt.imshow(
    display_image,
    extent=[-2.0, 1.0, -1.3, 1.3],
    origin="lower",
    cmap=cmap,
    vmin=0,
    vmax=max_iterations - 1
)

plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.title("Mandelbrot Set")

plt.colorbar(label="Escape iteration")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 10. Basic verification
# ---------------------------------------------------------

print("X shape:", X.shape)
print("c shape:", c.shape)
print("c dtype:", c.dtype)
print("escape_count shape:", escape_count.shape)
print("escape_count dtype:", escape_count.dtype)
print("Tensor device:", c.device)
print("Minimum escape count:", escape_count.min().item())
print("Maximum escape count:", escape_count.max().item())
```

### Important parts

* **Tensor shapes:** `X`, `Y`, `c`, `z`, `active`, and `escape_count` all represent the same (520\times600) grid. Therefore each tensor position `[row, column]` refers to one point in the complex plane.
* **Complex tensor:** `torch.complex(X, Y)` constructs (c=X+iY). Because `X` and `Y` are `float32`, `c` becomes a `torch.complex64` tensor.
* **Escape mask:** `active` starts as `True` everywhere. `escaped_now = active & (torch.abs(z) >= 4.0)` identifies only points that have crossed the escape radius **for the first time**, so their escape iteration is recorded once.
* **GPU usage:** `real`, `imag`, `X`, `Y`, `c`, `z`, the masks, and the iteration counts are all created on `device`. Consequently, when `device == cuda`, the entire 200-iteration Mandelbrot computation remains on the GPU. Only `escape_count.cpu().numpy()` transfers the final result to the CPU for Matplotlib.
* **Colour interpretation:** a small escape count means the point diverged quickly. A value of `200` means it remained bounded for all tested iterations; those points are rendered black.

The core computation can therefore be reduced conceptually to:

```text
c = x + yi
z₀ = 0

             ┌───────────────────────┐
             │  z = z² + c           │
             │  compute |z|          │
             │  detect new escapes   │
             │  record iteration     │
             └──────────┬────────────┘
                        │
                     × 200
                        │
                        ▼
              escape-count image
```

This is consistent with the official lab's Part 2 approach, which also uses a 2D complex tensor, repeatedly computes the Mandelbrot recurrence, creates a divergence mask from `torch.abs(...)`, and plots only after moving the final tensor back to the CPU. 

## Questin

## Prompt 2 — Create a high-resolution zoom of the Mandelbrot set

The previous PyTorch Mandelbrot code works.

Now modify it to demonstrate a high-resolution zoom into an interesting region of the Mandelbrot set, as required by my lab.

Requirements:

- Keep the computation in PyTorch and on CUDA GPU when available.
- Increase the grid resolution compared with the original version by decreasing the coordinate spacing.
- Zoom into an interesting boundary region of the Mandelbrot set rather than displaying the entire set.
- Choose a region where detailed fractal structures can clearly be seen.
- Use a resolution that gives a visibly high-quality image without using an unreasonable amount of GPU memory or computation time.
- Keep approximately 200–500 iterations.
- Print:
  - the device being used,
  - image/grid dimensions,
  - computation time.
- Explain which real-axis and imaginary-axis ranges you selected and why.
- Make it easy for me to change the zoom coordinates and resolution later.

Please give me the complete modified Python code and clearly identify the parameters I should change to experiment with different zoom levels.


## Answer

Yes. Your lab specifically requires a **higher-resolution Mandelbrot computation by decreasing the grid spacing and zooming into another region of the set**, while balancing image quality against computation time. 

For this example, I chose a boundary region around the well-known highly detailed area near

[
c\approx -0.744 + 0.132i
]

rather than displaying the entire Mandelbrot set.

```python
import time
import torch
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS TO CHANGE WHEN EXPERIMENTING
# ============================================================

# Zoom window
REAL_MIN = -0.76
REAL_MAX = -0.72

IMAG_MIN = 0.11
IMAG_MAX = 0.15

# Image resolution
# Original lab grid is roughly 600 x 520.
# This gives a much finer sampling of the zoomed region.
WIDTH = 1200
HEIGHT = 1200

# Mandelbrot parameters
MAX_ITERATIONS = 300
ESCAPE_RADIUS = 4.0


# ============================================================
# 1. SELECT CPU OR GPU
# ============================================================

# What: Use CUDA automatically when available.
# Why: The Mandelbrot calculation contains many independent
#      calculations that can be performed efficiently on a GPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. CREATE THE HIGH-RESOLUTION ZOOM GRID
# ============================================================

# What: Create real and imaginary coordinates directly
#       on the selected device.
# Why: This avoids unnecessary CPU -> GPU transfers.
real = torch.linspace(
    REAL_MIN,
    REAL_MAX,
    WIDTH,
    dtype=torch.float32,
    device=device
)

imag = torch.linspace(
    IMAG_MIN,
    IMAG_MAX,
    HEIGHT,
    dtype=torch.float32,
    device=device
)

# What: Turn the two coordinate vectors into a full 2D grid.
# Why: Every grid location represents one complex number c.
Y, X = torch.meshgrid(
    imag,
    real,
    indexing="ij"
)

print("Grid dimensions:", X.shape)

# Approximate coordinate spacing
real_spacing = (REAL_MAX - REAL_MIN) / (WIDTH - 1)
imag_spacing = (IMAG_MAX - IMAG_MIN) / (HEIGHT - 1)

print("Real-axis spacing:", real_spacing)
print("Imag-axis spacing:", imag_spacing)


# ============================================================
# 3. CREATE COMPLEX TENSORS
# ============================================================

# What: Construct c = x + yi.
# Why: Every pixel corresponds to one point in the complex plane.
c = torch.complex(X, Y)

# What: Set z_0 = 0 for every value of c.
# Why: This is the initial condition of the Mandelbrot iteration.
z = torch.zeros_like(c)

# What: Store the first iteration at which each point escapes.
# Why: This controls the colour of points outside the set.
escape_count = torch.full(
    c.shape,
    MAX_ITERATIONS,
    dtype=torch.int32,
    device=device
)

# What: Keep track of points that have not escaped yet.
# Why: We only want to record a point's first escape iteration.
active = torch.ones(
    c.shape,
    dtype=torch.bool,
    device=device
)


# ============================================================
# 4. START TIMER
# ============================================================

# CUDA operations are asynchronous.
# Synchronising ensures the timer measures the real GPU runtime.
if device.type == "cuda":
    torch.cuda.synchronize()

start_time = time.perf_counter()


# ============================================================
# 5. MANDELBROT COMPUTATION
# ============================================================

with torch.no_grad():

    for i in range(MAX_ITERATIONS):

        # What: Apply z_(n+1) = z_n^2 + c.
        # Why: This is the defining Mandelbrot recurrence.
        z = z * z + c

        # What: Find points that have just escaped.
        # Why: |z| >= 4 means the sequence is considered divergent.
        escaped_now = active & (
            torch.abs(z) >= ESCAPE_RADIUS
        )

        # Record their first escape iteration.
        escape_count[escaped_now] = i + 1

        # Remove escaped points from the active mask.
        active = active & (~escaped_now)


# ============================================================
# 6. STOP TIMER
# ============================================================

if device.type == "cuda":
    torch.cuda.synchronize()

end_time = time.perf_counter()

computation_time = end_time - start_time

print(f"Computation time: {computation_time:.4f} seconds")


# ============================================================
# 7. MOVE FINAL RESULT TO CPU
# ============================================================

# What: Transfer only the finished result.
# Why: No CPU/GPU transfers occurred inside the iteration loop.
mandelbrot = escape_count.cpu().numpy()


# ============================================================
# 8. PREPARE COLOURING
# ============================================================

display_image = mandelbrot.astype(float)

# Points that survive all iterations are treated as inside
# the Mandelbrot set and displayed as black.
display_image[mandelbrot == MAX_ITERATIONS] = -1

cmap = plt.get_cmap("turbo").copy()
cmap.set_under("black")


# ============================================================
# 9. DISPLAY THE HIGH-RESOLUTION ZOOM
# ============================================================

plt.figure(figsize=(10, 10))

plt.imshow(
    display_image,
    extent=[
        REAL_MIN,
        REAL_MAX,
        IMAG_MIN,
        IMAG_MAX
    ],
    origin="lower",
    cmap=cmap,
    vmin=0,
    vmax=MAX_ITERATIONS - 1,
    interpolation="nearest"
)

plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")

plt.title(
    "High-Resolution Mandelbrot Zoom\n"
    f"Real [{REAL_MIN}, {REAL_MAX}], "
    f"Imaginary [{IMAG_MIN}, {IMAG_MAX}]"
)

plt.colorbar(label="Escape iteration")

plt.tight_layout()
plt.show()
```

### Why this region?

The original lab covers approximately:

```text
Real:      -2.0  →  1.0
Imaginary: -1.3  →  1.3
```

This version instead examines:

```text
Real:      -0.76 → -0.72
Imaginary:  0.11 →  0.15
```

So the viewing window becomes dramatically smaller:

```text
Original view

-2 ------------------------------------ 1
        entire Mandelbrot set


Zoom

              -0.76 ---- -0.72
                    ↑
              detailed boundary
```

A Mandelbrot boundary is particularly useful because that is where you see the repeated fine-scale fractal structures. Areas deep inside the set would mostly appear black, while areas far outside would mostly escape immediately.

### Why is this higher resolution?

Your original lab grid uses approximately `0.005` coordinate spacing. 

Here, the real-axis spacing is approximately:

[
\frac{0.04}{1199}\approx 0.000033
]

So instead of moving roughly `0.005` between neighbouring samples, we're sampling at about `0.000033`.

That is roughly **150 times finer spatial sampling** within this particular coordinate window.

### The four parameters you should experiment with

You mainly need to modify this block:

```python
REAL_MIN = -0.76
REAL_MAX = -0.72

IMAG_MIN = 0.11
IMAG_MAX = 0.15

WIDTH = 1200
HEIGHT = 1200

MAX_ITERATIONS = 300
```

They control different things:

| Parameter              | Effect                                                   |
| ---------------------- | -------------------------------------------------------- |
| `REAL_MIN`, `REAL_MAX` | Horizontal location and zoom                             |
| `IMAG_MIN`, `IMAG_MAX` | Vertical location and zoom                               |
| `WIDTH`, `HEIGHT`      | Number of sampled points / image quality                 |
| `MAX_ITERATIONS`       | How long you test points before treating them as bounded |

The important distinction is:

```text
smaller coordinate range
        ↓
      ZOOM

more pixels in that range
        ↓
HIGHER RESOLUTION

more iterations
        ↓
better discrimination near the boundary
```

For example, to zoom further into the same area, you would change the **coordinate ranges**, rather than merely increasing `WIDTH` and `HEIGHT`.

One useful feature of this implementation is that when CUDA is available, `X`, `Y`, `c`, `z`, `active`, and `escape_count` all remain on the GPU throughout the iteration. The only substantial transfer back to the CPU happens here:

```python
mandelbrot = escape_count.cpu().numpy()
```

after the Mandelbrot computation has finished. This satisfies the lab's goal of using PyTorch parallelism for the actual fractal computation. 

## Question

## Prompt 3 — Convert the Mandelbrot code into a Julia set

Now modify the PyTorch Mandelbrot implementation so that it generates a Julia set instead.

For the Julia set, use

\[
z_{n+1}=z_n^2+c
\]

but unlike the Mandelbrot set:

- `z0` should be each point in the 2D complex-plane grid.
- `c` should be one fixed complex constant for the entire image.

Requirements:

- Use PyTorch complex tensors.
- Run the iterations on CUDA GPU if available.
- Use a 2D grid covering a suitable region such as real and imaginary values from approximately -1.5 to 1.5.
- Use approximately 200 iterations.
- Keep an escape/iteration counter so the Julia set can be coloured.
- Points that do not escape should be black.
- Use Matplotlib to display the result.
- Start with an interesting Julia constant such as `c = -0.8 + 0.156j`, but make `c` easy to change.
- Explain precisely what had to change compared with the Mandelbrot implementation, especially the roles of `z0` and `c`.

Please provide complete runnable code.

## Answer

Yes. This is the key conceptual change required for **COMP3710 Demo 1 Part 2.3**, where the lab asks you to modify the Mandelbrot implementation to produce a Julia set. 

For both sets, the recurrence is the same:

[
z_{n+1}=z_n^2+c
]

What changes is **what varies across the image**.

```text
Mandelbrot set
--------------
Each pixel → different c
z₀ = 0 for every pixel

Julia set
---------
Each pixel → different z₀
c = one fixed constant for every pixel
```

## Complete runnable PyTorch Julia-set code

```python
import torch
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS YOU CAN CHANGE
# ============================================================

# What: Fixed complex constant used for the entire Julia set.
# Why: Different values of c produce different Julia-set shapes.
C_REAL = -0.8
C_IMAG = 0.156

# What: Define the region of the complex plane used for z₀.
# Why: Each point in this region becomes one starting value z₀.
REAL_MIN = -1.5
REAL_MAX = 1.5

IMAG_MIN = -1.5
IMAG_MAX = 1.5

# What: Control image resolution.
# Why: More points produce a smoother and more detailed image.
WIDTH = 1000
HEIGHT = 1000

# What: Control how long each sequence is tested.
# Why: Points still bounded after all iterations are treated as part of the Julia set.
MAX_ITERATIONS = 200

# What: Define the divergence threshold.
# Why: Once |z| reaches this value, the sequence is treated as escaped.
ESCAPE_RADIUS = 4.0


# ============================================================
# 1. SELECT GPU OR CPU
# ============================================================

# What: Use CUDA automatically if it is available.
# Why: PyTorch can evaluate the whole grid in parallel on the GPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. CREATE THE 2D COMPLEX-PLANE GRID
# ============================================================

# What: Create real coordinates directly on the selected device.
# Why: These values form the real part of every starting point z₀.
real = torch.linspace(
    REAL_MIN,
    REAL_MAX,
    WIDTH,
    dtype=torch.float32,
    device=device
)

# What: Create imaginary coordinates directly on the selected device.
# Why: These values form the imaginary part of every starting point z₀.
imag = torch.linspace(
    IMAG_MIN,
    IMAG_MAX,
    HEIGHT,
    dtype=torch.float32,
    device=device
)

# What: Combine both coordinate vectors into two 2D grids.
# Why: We need every possible (real, imaginary) coordinate pair.
Y, X = torch.meshgrid(
    imag,
    real,
    indexing="ij"
)

print("Grid shape:", X.shape)


# ============================================================
# 3. CREATE z₀ AND FIXED c
# ============================================================

# What: Turn every grid position into a complex starting value.
# Why: In a Julia set, each pixel represents a different z₀.
z = torch.complex(X, Y)

# What: Create ONE fixed complex value c.
# Why: Unlike the Mandelbrot set, c does not vary from pixel to pixel.
c = torch.tensor(
    complex(C_REAL, C_IMAG),
    dtype=torch.complex64,
    device=device
)

print("z shape:", z.shape)
print("z dtype:", z.dtype)
print("Fixed c:", c)


# ============================================================
# 4. CREATE ESCAPE COUNTERS
# ============================================================

# What: Initially assume every point survives all iterations.
# Why: Points that never escape will retain MAX_ITERATIONS.
escape_count = torch.full(
    z.shape,
    MAX_ITERATIONS,
    dtype=torch.int32,
    device=device
)

# What: Keep track of points that have not escaped yet.
# Why: We want to record only the FIRST escape iteration.
active = torch.ones(
    z.shape,
    dtype=torch.bool,
    device=device
)


# ============================================================
# 5. JULIA-SET ITERATION
# ============================================================

with torch.no_grad():

    for i in range(MAX_ITERATIONS):

        # What: Apply z_(n+1) = z_n² + c.
        # Why: This recurrence determines whether each initial z₀
        #      remains bounded or diverges.
        z_new = z * z + c

        # What: Determine which active points escape now.
        # Why: Their magnitude has reached the escape radius.
        escaped_now = active & (
            torch.abs(z_new) >= ESCAPE_RADIUS
        )

        # What: Record their first escape iteration.
        # Why: Escape speed will determine the displayed colour.
        escape_count[escaped_now] = i + 1

        # What: Remove newly escaped points from the active mask.
        # Why: Their escape time has already been recorded.
        active = active & (~escaped_now)

        # What: Update only points that are still active.
        # Why: This prevents already-diverged values from growing
        #      unnecessarily large.
        z = torch.where(active, z_new, z)


# ============================================================
# 6. MOVE FINAL RESULT BACK TO CPU
# ============================================================

# What: Transfer the completed escape-count image to the CPU.
# Why: Matplotlib operates on NumPy/CPU data.
julia = escape_count.cpu().numpy()


# ============================================================
# 7. MAKE NON-ESCAPING POINTS BLACK
# ============================================================

display_image = julia.astype(float)

# What: Give non-escaping points a value below the colour range.
# Why: cmap.set_under() will then display those points as black.
display_image[julia == MAX_ITERATIONS] = -1

cmap = plt.get_cmap("turbo").copy()
cmap.set_under("black")


# ============================================================
# 8. DISPLAY THE JULIA SET
# ============================================================

plt.figure(figsize=(10, 10))

plt.imshow(
    display_image,
    extent=[
        REAL_MIN,
        REAL_MAX,
        IMAG_MIN,
        IMAG_MAX
    ],
    origin="lower",
    cmap=cmap,
    vmin=0,
    vmax=MAX_ITERATIONS - 1,
    interpolation="nearest"
)

plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")

plt.title(
    f"Julia Set: c = {C_REAL} + {C_IMAG}j"
)

plt.colorbar(label="Escape iteration")

plt.tight_layout()
plt.show()


# ============================================================
# 9. BASIC VERIFICATION
# ============================================================

print("Image dimensions:", julia.shape)
print("Escape-count dtype:", escape_count.dtype)
print("Tensor device:", escape_count.device)
print("Minimum escape count:", escape_count.min().item())
print("Maximum escape count:", escape_count.max().item())
```

## Precisely what changed from Mandelbrot?

This is the most important part to understand for your demo.

### 1. Mandelbrot: the grid represents `c`

In Mandelbrot:

```python
c = torch.complex(X, Y)
z = torch.zeros_like(c)
```

That means every pixel gets a different (c):

[
c_{ij}=x_{ij}+iy_{ij}
]

but every sequence starts from

[
z_0=0.
]

So:

```text
pixel 1 → c₁ → z₀ = 0
pixel 2 → c₂ → z₀ = 0
pixel 3 → c₃ → z₀ = 0
...
```

---

### 2. Julia: the grid represents `z₀`

For Julia, this becomes:

```python
z = torch.complex(X, Y)
```

Now every pixel itself is a different starting point:

[
z_0=x+iy.
]

Meanwhile:

```python
c = torch.tensor(
    complex(C_REAL, C_IMAG),
    dtype=torch.complex64,
    device=device
)
```

creates **one fixed (c)** shared by every point.

```text
pixel 1 → z₀₁ ─┐
pixel 2 → z₀₂ ─┼─→ same c = -0.8 + 0.156i
pixel 3 → z₀₃ ─┤
pixel 4 → z₀₄ ─┘
```

The recurrence itself has **not changed**:

```python
z_new = z * z + c
```

That is still

[
z_{n+1}=z_n^2+c.
]

The difference is entirely in the roles of the grid and (c).

| Property        | Mandelbrot                    | Julia                           |
| --------------- | ----------------------------- | ------------------------------- |
| Grid represents | (c)                           | (z_0)                           |
| (z_0)           | Always 0                      | Different at each pixel         |
| (c)             | Different at each pixel       | One fixed constant              |
| Iteration       | (z^2+c)                       | (z^2+c)                         |
| Output          | Which (c) values stay bounded | Which (z_0) values stay bounded |

That distinction is the core mathematical idea behind converting the lab's Mandelbrot code into a Julia-set implementation. The official lab explicitly requires modifying the Part 2 code to show a Julia set. 

### Changing the Julia set

The easiest parameters to experiment with are:

```python
C_REAL = -0.8
C_IMAG = 0.156
```

For example, another constant would simply be:

```python
C_REAL = -0.4
C_IMAG = 0.6
```

You do **not** need to change the algorithm. Changing (c) changes the dynamical system itself, which can produce a dramatically different Julia-set structure.

### Quick self-check

With CUDA available, you should see something similar to:

```text
Device: cuda
Grid shape: torch.Size([1000, 1000])
z shape: torch.Size([1000, 1000])
z dtype: torch.complex64
Fixed c: tensor(-0.8000+0.1560j, device='cuda:0')
Tensor device: cuda:0
```

The most useful one-sentence explanation for a demonstrator is:

> **“For the Mandelbrot set, each pixel is a different value of (c) and (z_0=0); for the Julia set, each pixel is a different initial value (z_0), while (c) is fixed for the entire image.”**


## link

https://chatgpt.com/share/6a862bd1-4b6c-83ec-a9a9-551be06e002e
