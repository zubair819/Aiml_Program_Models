import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

#define the data points
points=np.array([[3,7], [4,6], [5,5], [6,4], [7,3], [6,2], [7,2], [8,4], [3,3], [2,6], [3,5], [2,4]])

#plot the data points
plt.figure(figsize=(6,6))
plt.scatter(points[:,0], points[:,1], color='b')
plt.title("Raw Data Point")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#apply DBSCAN
db = DBSCAN(eps=1.9, min_samples=4).fit(points)
labels = db.labels_

#Extract core points, border points, and noise
core_points_mask=np.zeros_like(labels, dtype=bool)
core_points_mask[db.core_sample_indices_]=True
border_points_mak = (labels != -1) & ~core_points_mask
noise_points_mask=(labels == -1)

#plot core points, border points, and noise
plt.figure(figsize=(6,6))
plt.scatter(points[core_points_mask, 0], points[core_points_mask, 1], color='red', marker='o', label='Core points')
plt.scatter(points[border_points_mak, 0], points[border_points_mak, 1 ], color='green', marker='o', label='Border points')
plt.scatter(points[noise_points_mask, 0], points[noise_points_mask, 1], color='black', marker='o', label='Noise points')

#plot epsilon neighborhoods around coe points as circles
for point in points[core_points_mask]:
    circle = plt.Circle((point[0], point[1]), 1.9, color='blue', fill=False, linestyle='dotted')
    plt.gca().add_artist(circle)
    
plt.title("DBSCAN Clustering with Approximate Boundaries")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal") #Ensure equal scaling fr x and y axis
plt.show()
print("DBSCAN labels: ", labels) #Print the labels assigned by DBSCAN