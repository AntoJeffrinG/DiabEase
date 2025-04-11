import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

# Paths to the dataset folders (update this with your correct paths)
DR_images = []
No_DR_images = []

# Load "DR" images (Diabetic Retinopathy)
for f in glob.iglob("DIABESE_MODEL/data/Diagnosis of Diabetic Retinopathy/train/DR/*.jpg"):
    img = cv2.imread(f)
    img = cv2.resize(img, (128, 128))  # Resize to 128x128
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])  # Convert to RGB
    DR_images.append(img)

# Load "No_DR" images (No Diabetic Retinopathy)
for f in glob.iglob("DIABESE_MODEL/data/Diagnosis of Diabetic Retinopathy/train/No_DR/*.jpg"):
    img = cv2.imread(f)
    img = cv2.resize(img, (128, 128))  # Resize to 128x128
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])  # Convert to RGB
    No_DR_images.append(img)

# Convert lists to numpy arrays
DR_images = np.array(DR_images)
No_DR_images = np.array(No_DR_images)

print("DR images shape: ", DR_images.shape)
print("No_DR images shape: ", No_DR_images.shape)

# Visualize random images from both categories
def plot_random(DR_images, No_DR_images, num=5):
    # Select random images
    DR_imgs = DR_images[np.random.choice(DR_images.shape[0], num, replace=False)]
    No_DR_imgs = No_DR_images[np.random.choice(No_DR_images.shape[0], num, replace=False)]
    
    # Plot DR images
    plt.figure(figsize=(16, 9))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.title('DR')
        plt.imshow(DR_imgs[i])

    # Plot No_DR images
    plt.figure(figsize=(16, 9))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.title('No_DR')
        plt.imshow(No_DR_imgs[i])

# Call the function to visualize
plot_random(DR_images, No_DR_images, num=5)
plt.show()
