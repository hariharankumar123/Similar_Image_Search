# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:15:17 2023

@author: iamha
"""

#%%


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import mahotas.features
from scipy.fftpack import dct
from scipy.spatial.distance import euclidean
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image



#%%   Read the DataSet

dataset_folder = 'C:/Users/iamha/Downloads/Annotation_removed'
image_files = os.listdir(dataset_folder)

#%%   2D Detection of the Data set


lower_limit = 5
upper_limit = 200

all_shapes = []

all_masks = []

# Process each image
for filename in image_files:
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        image_path = os.path.join(dataset_folder, filename)
        image = cv2.imread(image_path, 0)

        # Detect shapes along rows
        row_shapes = []
        rows, cols = image.shape
        for row in range(rows):
            row_intensities = image[row, :]
            row_mask = np.logical_and(row_intensities >= lower_limit, row_intensities <= upper_limit)
            row_diff = np.diff(row_mask.astype(np.int32))
            row_starts = np.where(row_diff == 1)[0]
            row_ends = np.where(row_diff == -1)[0]
            for i in range(len(row_starts)):
                if i < len(row_ends) and row_ends[i] - row_starts[i] + 1 >= 10:
                    row_shapes.append((row, row_starts[i], row, row_ends[i]))
    
        # Detect shapes along columns
        col_shapes = []
        for col in range(cols):
            col_intensities = image[:, col]
            col_mask = np.logical_and(col_intensities >= lower_limit, col_intensities <= upper_limit)
            col_diff = np.diff(col_mask.astype(np.int32))
            col_starts = np.where(col_diff == 1)[0]
            col_ends = np.where(col_diff == -1)[0]
            for i in range(len(col_starts)):
                if i < len(col_ends) and col_ends[i] - col_starts[i] + 1 >= 10:
                    col_shapes.append((col_starts[i], col, col_ends[i], col))

        # Determine the minimum number of shapes for rows and columns
        min_shapes = min(len(row_shapes), len(col_shapes))

        # Combine row and column shapes
        shapes = row_shapes[:min_shapes] + col_shapes[:min_shapes]

        # Store the detected shapes for the current image
        all_shapes.append(shapes)
        

    
#%%       DISPLAY THE DATASET
    
    

for idx, filename in enumerate(image_files):
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        image_path = os.path.join(dataset_folder, filename)
        image = cv2.imread(image_path)

        # Create a mask image with black pixels
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw bounding boxes around the detected shapes on the mask image
        for shape in all_shapes[idx]:
            x1, y1, x2, y2 = shape
            cv2.rectangle(mask, (y1, x1), (y2, x2), 255, cv2.FILLED)  

        # Invert the mask image
        mask = cv2.bitwise_not(mask)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Display the images
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Shape Detection Result')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')  
        plt.title('Mask Image')
        plt.axis('off')
        plt.show()
        
        
#%%    QUERY IMAGE


query_image = cv2.imread('C:/Users/iamha/Downloads/Annotation_removed/36432838.jpeg', 0)

lower_limit = 5
upper_limit = 200

query_all_shapes = []

# Detect shapes along rows
query_row_shapes = []
query_rows, query_cols = query_image.shape
for query_row in range(query_rows):
    query_row_intensities = query_image[query_row, :]
    query_row_mask = np.logical_and(query_row_intensities >= lower_limit, query_row_intensities <= upper_limit)
    query_row_diff = np.diff(query_row_mask.astype(np.int32))
    query_row_starts = np.where(query_row_diff == 1)[0]
    query_row_ends = np.where(query_row_diff == -1)[0]
    for query_i in range(len(query_row_starts)):
        if query_i < len(query_row_ends) and query_row_ends[query_i] - query_row_starts[query_i] + 1 >= 10:
            query_row_shapes.append((query_row, query_row_starts[query_i], query_row, query_row_ends[query_i]))


# Detect shapes along columns
query_col_shapes = []
for query_col in range(query_cols):
    query_col_intensities = query_image[:, query_col]
    query_col_mask = np.logical_and(query_col_intensities >= lower_limit, query_col_intensities <= upper_limit)
    query_col_diff = np.diff(query_col_mask.astype(np.int32))
    query_col_starts = np.where(query_col_diff == 1)[0]
    query_col_ends = np.where(query_col_diff == -1)[0]
    for query_i in range(len(query_col_starts)):
        if query_i < len(query_col_ends) and query_col_ends[query_i] - query_col_starts[query_i] + 1 >= 10:
            query_col_shapes.append((query_col_starts[query_i], query_col, query_col_ends[query_i], query_col))


# Determine the minimum number of shapes for rows and columns
query_min_shapes = min(len(query_row_shapes), len(query_col_shapes))

# Combine row and column shapes
query_shapes = query_row_shapes[:query_min_shapes] + query_col_shapes[:query_min_shapes]

# Store the detected shapes for the current image
query_all_shapes.append(query_shapes)


# Create a binary masked image
query_mask = np.zeros_like(query_image, dtype=np.uint8)
for shape in query_row_shapes:
    x1, y1, x2, y2 = shape
    query_mask[x1:x2+1, y1:y2+1] = 255
for shape in query_col_shapes:
    x1, y1, x2, y2 = shape
    query_mask[x1:x2+1, y1:y2+1] = 255
query_mask = 255 - query_mask


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
output_image = query_image.copy()
for shape in query_row_shapes:
    x1, y1, x2, y2 = shape
    cv2.rectangle(output_image, (y1, x1), (y2, x2), (0, 255, 0), 2)
for shape in query_col_shapes:
    x1, y1, x2, y2 = shape
    cv2.rectangle(output_image, (y1, x1), (y2, x2), (0, 0, 255), 2)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB))
plt.title('Detected Shapes')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(query_mask, cmap='gray')
plt.title('Binary Masked Image')
plt.axis('off')

plt.tight_layout()
plt.show()


#%%    SIMILAR IMAGE SEARCH


# Load the pretrained VGG16 model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Resize and preprocess the query image
query_image = cv2.imread('C:/Users/iamha/Downloads/Annotation_removed/36432838.jpeg')
query_image = cv2.resize(query_image, (224, 224))
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
query_image = np.expand_dims(query_image, axis=0)
query_image = preprocess_input(query_image)

# Extract features for the query image
query_features = model.predict(query_image)

# Compare query features with dataset images
dataset_folder = 'C:/Users/iamha/Downloads/Annotation_removed'
dataset_images = os.listdir(dataset_folder)

threshold = 90

similarity_list = []

for dataset_image_name in dataset_images:
    dataset_image_path = os.path.join(dataset_folder, dataset_image_name)
    dataset_image = Image.open(dataset_image_path)
    dataset_image = dataset_image.resize((224, 224))
    dataset_image = dataset_image.convert('RGB')
    dataset_image = np.array(dataset_image)
    dataset_image = np.expand_dims(dataset_image, axis=0)
    dataset_image = preprocess_input(dataset_image)

    dataset_features = model.predict(dataset_image)

    similarity = 1 - cosine(query_features.flatten(), dataset_features.flatten())

    similarity_percentage = similarity * 100

    similarity_list.append((similarity_percentage, dataset_image_path))

similarity_list.sort(reverse=True)

top_3_similar_images = []
for similarity_percentage, image_path in similarity_list:
    if similarity_percentage > threshold:
        top_3_similar_images.append((similarity_percentage, image_path))

# Display the top 3 similar images
for similarity_percentage, image_path in top_3_similar_images[:3]:
    dataset_image = Image.open(image_path).convert('RGB')
    plt.imshow(dataset_image)
    plt.axis('off')
    plt.title(f"Similarity: {similarity_percentage:.2f}%")
    
    # Extract the file name (image name) from the image path
    file_name = os.path.basename(image_path)
    
    # Remove file extension from the image name
    image_name = os.path.splitext(file_name)[0]
    
    # Display the image name
    print("Image Name:", image_name)
    
    plt.show()


#%%