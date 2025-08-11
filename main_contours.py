import cv2                    # For image processing
import numpy as np           # For array manipulation
from skimage.metrics import structural_similarity as ssim  # SSIM comparison
from noise import pnoise2    # Perlin noise generation
import matplotlib.pyplot as plt  # For visualization
import os                    # To confirm working directory
from dotenv import load_dotenv # import env retrieval
load_dotenv() # actually load env variables

IMAGE_DIR = os.getenv("IMAGE_DIR", "images")  # Default fallback


# Print the current working directory to confirm file paths
print("Current Working Directory:", os.getcwd())

# --- Load Cyanotype Image ---
def load_image(path, size=(512, 512)):
    """
    Loads and resizes an image from a given path. Converts it to float32 in [0, 1].
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    img = cv2.Canny(img, 100, 200) # We want to define thresholds before resizing to match Perlin noise
    img = cv2.resize(img, size)                   # Resize to match Perlin noise shape
    img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return img

# --- Generate Perlin Noise ---
def generate_perlin(size=(512, 512), scale=10):
    """
    Creates a 2D Perlin noise array scaled to match the image (512 x 512 pixels)
    """
    width, height = size
    noise_map = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            noise_map[i][j] = pnoise2(i / scale, j / scale, octaves=6)
    # Normalize to [0, 1]
    noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
    return noise_map

# --- Compare Two Images with SSIM ---
def compare_images(img1, img2):
    """
    Calculates SSIM similarity score and difference map.
    """
    score, diff = ssim(img1, img2, full=True, data_range=1.0)
    print(f"SSIM Similarity Score: {score:.4f}")
    return diff

# --- Plot Results ---
def plot_images(img1, img2, diff):
    """
    Display the cyanotype, Perlin noise, and SSIM difference map.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img1, cmap='gray')
    axs[0].set_title("Cyanotype")
    axs[1].imshow(img2, cmap='gray')
    axs[1].set_title("Perlin Noise")
    axs[2].imshow(diff, cmap='hot')
    axs[2].set_title("SSIM Difference")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/ssim_comparison.png")  # Save the visualization
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    cyanotype_img = load_image("images/cyanotype.jpg")
    perlin_img = generate_perlin()
    diff_map = compare_images(cyanotype_img, perlin_img)
    plot_images(cyanotype_img, perlin_img, diff_map)