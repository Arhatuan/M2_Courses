import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

GRAY_FILENAME = "milky-way-nvg.jpg"
COLOR_FILENAME = "milky-way.jpg"
    
def fuzzyCmeans(X, img, c = 2, m = 2, max_iter = 5, epsilon = 0.001):
    """Fuzzy C-Means implementation for an image
    
    Arguments:
        X (Matrix-like) : image's data flattened to a 1D vector for each channel
        img : original image 
        c (int) : number of cluster (defaults to 2).
        m (float) : fuzziness coefficient (defaults to 2).
        max_iter (int) : max number of iterations (defaults to 5).
        epsilon (float) : convergence threshold (defaults to 0.001).
        color (bool) : if False, use grayscale image, if True, use colored image (defaults to False).
        rescale (float) : rescale factor to resize the image (defaults to 1).
    """
    assert c >= 2, "Number of clusters must be 2 or more."
    assert m >= 1, "Fuziness coefficient must be any float >= 1"

    # Initialize membership matrix, then normalize the values
    U = np.random.rand(X.shape[0], c)
    U = U / np.sum(U, axis=0)

    for iteration in range(1, max_iter+1):
        timeStart = time.time()

        # Calculate centers, update membership matrix, compute membership difference
        centers = calculate_centers(X, U, m)
        U_new = update_membership(X, U, centers, m)
        diff_U = np.linalg.norm(U_new - U)
        U = U_new

        timeEnd = time.time()
        print(f"end iteration {iteration} (difference : {diff_U :.4f}) (time : {timeEnd - timeStart :.2f}s)")
        
        if diff_U < epsilon:
            print("Converged at iteration :", iteration)
            break
    
    # print("\nFinal cluster centers : ")
    # print(centers)

    display_segmented_img(c, U, centers, X, img, iteration)



#######################
## CENTERS calculation
#######################
def calculate_centers(X, U, m):
    """Calculte centers from the given U membership matrix and X values"""
    um = U**m
    centers = []

    for j in range(U.shape[1]):
        sum1 = 0
        for i in range(U.shape[0]):
            sum1 += um[i,j] * X[i]

        sum2 = np.sum(um[:,j])
        
        centers.append(sum1 / sum2)

    return centers

#######################
## UPDATE membership
#######################
def update_membership(X, U, centers, m):
    """Update the membership matrix U, based on given centers and X values"""
    power = 2 / (m-1)
    U_new = np.zeros_like(U)

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            sum = 0
            for k in range(len(centers)):
                d1 = np.linalg.norm(X[i] - centers[j])
                d2 = np.linalg.norm(X[i] - centers[k])
                sum += (d1 / d2) ** power
            U_new[i,j] = 1 / sum

    return U_new


############################
## Segmentation and display
############################
def display_segmented_img(c, U, centers, X, img, nb_iter):
    labels = list(range(len(centers)))
    segmented = np.zeros(X.shape[0])
    class_stats = np.zeros(len(centers))

    for i in range(X.shape[0]):
        # assign class = center of cluster (chosen by max coefficient)
        max_coeff_id = np.argmax(U[i])
        segmented[i] = labels[max_coeff_id] #centers[max_coeff_id]
        class_stats[max_coeff_id] += 1

    # Reshape the segmentation vector to the original image size
    segmented = segmented.reshape(img.shape[:2]).astype(np.uint8)

    print("\nClasses stats :")
    proportions = []
    for j in range(len(centers)):
        proportion = class_stats[j] / np.sum(class_stats)
        print(f"\t - Class {labels[j]}  (center {str(centers[j])}) : {proportion :.2%}")
        proportions.append(proportion)

    plt.figure(figsize=(15,5))
    plt.suptitle(f"Fuzzy C-Means ({c} classes, {nb_iter} iterations)")
    nb_cols = max(3, U.shape[1])

    # 1) Original image
    plt.subplot(2, nb_cols, 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Original image ({img.shape[0]}x{img.shape[1]})")
    plt.axis("off")

    # 2) Segmentation (on maximum membership coefficient)
    cmap = plt.get_cmap('tab10', c)
    plt.subplot(2, nb_cols, 2)
    plt.imshow(segmented, cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(ticks=range(c))
    cbar.ax.set_yticklabels([f'Cl. {i} ({proportions[i]:.1%})' for i in range(c)])
    plt.title("Fuzzy C-Means Segmentation")
    plt.axis("off")

    # 3) Heatmap (maximum membership coefficient)
    membership_map = np.max(U, axis=1).reshape(img.shape[:2])
    plt.subplot(2, nb_cols, 3)
    plt.imshow(membership_map, cmap='terrain_r', vmin=0, vmax=1)
    plt.colorbar(label="Max membership coeff")
    plt.title("Maximum membership")
    plt.axis("off")

    # 4) Heatmaps (membership coefficient for each class)
    line2_firstId = max(4, nb_cols+1)
    for j in range(U.shape[1]):
        class_membership = U[:,j].reshape(img.shape[:2])
        plt.subplot(2, nb_cols, line2_firstId + j)
        plt.imshow(class_membership, cmap='coolwarm', vmin=0, vmax=1)
        plt.colorbar(label=f"Class {j} membership coeff")
        plt.title(f"Class {j} membership")
        plt.axis("off")

    plt.show()


def manager_fuzzyCmeans(c = 2, m = 2, max_iter = 5, epsilon = 0.001, color = False, rescale = 1):
    """Manager for the Fuzzy C-Means implementation (select image, resize it, generate vector X)
        
    Arguments:
        c (int) : number of cluster (defaults to 2).
        m (float) : fuzziness coefficient (defaults to 2).
        max_iter (int) : max number of iterations (defaults to 5).
        epsilon (float) : convergence threshold (defaults to 0.001).
        color (bool) : if False, use grayscale image, if True, use colored image (defaults to False).
        rescale (float) : rescale factor to resize the image (defaults to 1).
    """
    
    # Choose the image (gray or colored)
    if not color:
        img = cv2.imread(GRAY_FILENAME, flags=cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, None, fx=rescale, fy=rescale)
        X = img.reshape(-1) # reshape to 1D vector
        X = np.float64(X)
    else:
        img = cv2.imread(COLOR_FILENAME)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, None, fx=rescale, fy=rescale)
        X = img.reshape((-1,3)) # reshape to 1D vectors for each channel
        X = np.float64(X)

    fuzzyCmeans(X, img, c=c, m=m, max_iter=max_iter, epsilon=epsilon)


if __name__ == "__main__":
    
    manager_fuzzyCmeans(
        c = 2,
        m = 2,
        max_iter = 30,
        epsilon = 0.001,
        color = True,
        rescale = 0.25
    )