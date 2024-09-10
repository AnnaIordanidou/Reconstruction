import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

mnist = fetch_openml('mnist_784', version=1, parser = 'auto')
data, target = mnist["data"], mnist["target"]

avg_digit = {}
for digit in ['1', '9']:
    digit_data = data[target == digit]
    avg_digit[digit] = digit_data.mean(axis=0)


plt.imshow(avg_digit['1'].values.reshape(28, 28), cmap='gray')
plt.title('Average Digit for 1')
plt.show()

plt.imshow(avg_digit['9'].values.reshape(28, 28), cmap='gray')
plt.title('Average Digit for 9')
plt.show()


cov_matrices = {}
for digit in ['1', '9']:
    digit_data = data[target == digit]

    mean_vector = np.mean(digit_data, axis=0)
    centered_data = digit_data - mean_vector

    cov_matrix = (1 / (len(centered_data) - 1)) * np.dot(centered_data.T, centered_data)

    cov_matrices[digit] = cov_matrix

cov_matrix_for_1 = cov_matrices['1']

cov_matrix_for_9 = cov_matrices['9']

for digit, cov_matrix in cov_matrices.items():
    plt.figure(figsize=(10, 10))
    plt.imshow(cov_matrix, cmap='hot', interpolation='nearest')
    plt.title(f'Covariance Matrix for Digit {digit}')
    plt.colorbar()
    plt.show()


n_components = 8
pca = PCA(n_components=n_components)
principal_components = {}

for digit in ['1', '9']:
    digit_data = data[target == digit]
    principal_components[digit] = pca.fit_transform(digit_data)


reconstructed_digits = {}

for L in [1, 8, 16, 64, 256]:
    pca = PCA(n_components=L)
    reconstructed_digits[L] = {}
    for digit in ['1', '9']:
        digit_data = data[target == digit]
        principal_components_temp = pca.fit_transform(digit_data)
        reconstructed = pca.inverse_transform(principal_components_temp)
        reconstructed_digits[L][digit] = reconstructed


errors = {L: {} for L in [1, 8, 16, 64, 256]}

for L, reconstructions in reconstructed_digits.items():
    for digit in ['1', '9']:
        digit_data = data[target == digit]
        error = np.linalg.norm(digit_data - reconstructions[digit], axis=1)
        errors[L][digit] = error


for L in [1, 8, 16, 64, 256]:
    for digit in ['1', '9']:
        plt.figure(figsize=(10, 5))
        plt.hist(errors[L][digit], bins=50, alpha=0.6, label=f'Errors for {digit} with L={L}')
        plt.title(f'Histogram of Reconstruction Errors for {digit} with L={L}')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


for L, reconstructions in reconstructed_digits.items():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(reconstructions['1'][0].reshape(28, 28), cmap='gray')
    plt.title(f'Reconstructed 1 for L={L}')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructions['9'][0].reshape(28, 28), cmap='gray')
    plt.title(f'Reconstructed 9 for L={L}')

    plt.tight_layout()
    plt.show()

