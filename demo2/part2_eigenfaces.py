import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# COMP3710 Demo 2 - Part 2: Eigenfaces
# Q1: Load the LFW face dataset and inspect its structure.


# [Provided] Load the Labeled Faces in the Wild dataset.
# Keep people with at least 70 images and resize images to 40% of their original size.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)


# [Provided] Extract the image dimensions.
# n_samples = number of face images; h = image height; w = image width.
n_samples, h, w = lfw_people.images.shape


# [Provided] Flatten each h × w face image into one feature vector.
# X.shape = (number of images, number of pixels per image).
X = lfw_people.data


# [Provided] Number of input features (pixels) for each face.
n_features = X.shape[1]


# [Provided] Integer class label for each face image.
y = lfw_people.target


# [Provided] Person names corresponding to the integer class labels.
target_names = lfw_people.target_names


# [Provided] Number of different people/classes in the dataset.
n_classes = target_names.shape[0]


print("\n--- LFW Dataset ---")
print("n_samples:", n_samples)
print("image shape:", (h, w))
print("X shape:", X.shape)
print("y shape:", y.shape)
print("n_features:", n_features)
print("n_classes:", n_classes)

# Q2 | Official: split the dataset into training and testing sets to avoid model contamination.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\n--- Train/Test Split ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3 | Official: center the training data, compute PCA using SVD, and project faces into face space.

n_components = 150  # Number of principal components/eigenfaces to keep.

# Compute the mean face using training data only.
mean = np.mean(X_train, axis=0)

# Center both datasets using the training-set mean.
X_train_centered = X_train - mean
X_test_centered = X_test - mean

# Compute the SVD of the centered training data.
U, S, V = np.linalg.svd(X_train_centered, full_matrices=False)

# Keep the first 150 principal directions.
components = V[:n_components]

# Reshape each principal component back into a 50 × 37 image.
eigenfaces = components.reshape((n_components, h, w))

# ★ CORE: Project each face from 1850-dimensional pixel space into 150-dimensional PCA face space.
X_transformed = np.dot(X_train_centered, components.T)
X_test_transformed = np.dot(X_test_centered, components.T)

print("\n--- PCA / Face Space ---")
print("Mean shape:", mean.shape)
print("Components shape:", components.shape)
print("Eigenfaces shape:", eigenfaces.shape)
print("X_train transformed:", X_transformed.shape)
print("X_test transformed:", X_test_transformed.shape)


# Q4 | Official: plot the principal components as eigenfaces.

OUTPUT_DIR = Path(__file__).resolve().parent  # Save figures beside this script.


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Plot a gallery of face-like images.

    Use: plot_gallery(images, titles, h, w, n_row, n_col)
    images: array containing flattened or 2D face images.
    titles: title for each displayed image.
    h, w: face image height and width.
    n_row, n_col: number of rows and columns in the gallery.
    """
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))

    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)

        # ★ CORE: reshape one PCA component back into image form.
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)

        plt.title(titles[i], size=10)
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()
    return fig


# Create titles for the first eigenfaces.
eigenface_titles = [f"Eigenface {i}" for i in range(eigenfaces.shape[0])]

# ★ CORE: visualise the first 12 principal components.
fig = plot_gallery(
    eigenfaces,
    eigenface_titles,
    h,
    w,
    n_row=3,
    n_col=4
)

eigenfaces_path = OUTPUT_DIR / "eigenfaces.png"
fig.savefig(eigenfaces_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# Q5 | Official: evaluate PCA dimensionality reduction using a compactness plot.

# Compute the variance represented by each singular value.
explained_variance = (S ** 2) / (n_samples - 1)

# Convert each component's variance into a proportion of the total variance.
total_var = explained_variance.sum()
explained_variance_ratio = explained_variance / total_var

# ★ CORE: cumulative variance shows how much information is retained
# when the first 1, 2, 3, ... principal components are kept.
ratio_cumsum = np.cumsum(explained_variance_ratio)

# Create x-axis values for the first 150 PCA components.
eigenvalue_count = np.arange(n_components)

# Plot cumulative explained variance.
fig = plt.figure(figsize=(8, 5))
plt.plot(eigenvalue_count, ratio_cumsum[:n_components])

plt.title("PCA Compactness")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()

# Save the compactness plot beside this script.
compactness_path = OUTPUT_DIR / "pca_compactness.png"
fig.savefig(compactness_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# Print the key value needed to interpret the chosen 150 components.
print(
    "\nCumulative explained variance with "
    f"{n_components} components: "
    f"{ratio_cumsum[n_components - 1]:.4f}"
)

# Q6 | Official: classify the PCA face-space features using a Random Forest.

# Build the Random Forest using the parameters provided in the lab sheet.
estimator = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    max_features=150
)

# ★ CORE: learn the mapping from 150 PCA features to the person labels.
estimator.fit(X_transformed, y_train)

# ★ CORE: predict the identities of the unseen test faces.
predictions = estimator.predict(X_test_transformed)

# Compare predictions with the true test labels.
correct = predictions == y_test
total_test = len(y_test)
accuracy = np.sum(correct) / total_test

print("\n--- Random Forest Classification ---")
print("Total testing:", total_test)
print("Total correct:", np.sum(correct))
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=target_names,
        zero_division=0
    )
)