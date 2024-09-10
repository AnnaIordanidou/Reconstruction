```
• Using Numpy's mean function to calculate the middle digit of two digits.
• The function averages the values of each pixel position in the images.
• The process is repeated for all pixel positions to arrive at a final vector.
• The vector is converted into a 28x28 matrix to represent the average pixel intensity for a particular location.
• The covariance register is calculated, showing the relationships between each pixel and every other pixel for that particular digit.
• PCA is used to calculate the first 8 principal components, identifying the principal components where the data differ most.
• After dimensionality reduction, the original data is reconstructed from its compressed form using a lower dimensional representation and projecting it back into the original space.
• Calculated by calculating Euclidean distance of original digits to reconstructed ones.
• Measures information loss during dimensionality reduction and reconstruction.
• Visualized using histograms of digits for different L.
```

**Observations:**
 1) With a larger number of components, i.e. larger L values, the reconstructed image is closer to the original. This is evident in the visualization of the reconstructed digits, where with higher L values, the images are clearer.


2) The histograms illustrate how the reconstruction accuracy varies with different values ​​of L. The height of the histogram shows the frequency of these errors. A narrower spread indicates that most reconstructions have similar error, while a wider one indicates more variability in the quality of the reconstruction at different data points. A lower error indicates a better reconstruction. As L increases, the error distributions tend to be narrower, indicating more accurate reconstructions.

