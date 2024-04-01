## Ver1.0, by S. S. Arslan, M. Fux, P. Sinha, 2024.
## Please see the reference paper for the description of parameters: https://www.biorxiv.org/content/10.1101/2023.08.03.551725v1

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import matplotlib.pyplot as plt  # Add this line to import matplotlib
import tensorflow as tf

from sympy import symbols, solve # Might need to install/set up 'sympy'

def get_eff_resolution(interpupildist_inch, interpupildist_px, distance):
    
    # Your code to calculate eff_res for the given distance
    Deg_max = 80  # maximum degree the angular resolution is defined.
    S = 62  # peripheral roll-off "slope" in terms of cpd/o. 55-75 are based on the analysis of the historical data.
    Vaccuity = 20  # visual acuity: Ex: 20/Vaccuity=20 for normal vision.
    Deg_max = 80  # maximum degree the angular resolution is defined.
    fov_lin = 0  # 1:linear, 0:non-linear cone distribution inside the foveal zone, particularly at the pit.
    e_0 = 1.8  # this is a threshold parameter below which fovea density is assumed to be the same or extremely nonlinear w cones.
   
    interpupildist_inch /= 12  # in terms of foot.
    angular_resolution = 30 * (20 / Vaccuity)  # in terms of cpd: cycle per degree.
    B = 2.1 * (20 / Vaccuity)
    
    x = symbols('x')
    Fv = float(solve(B ** (S / angular_resolution) - B ** (S / (30 * (20 / x))) - e_0, x)[0])  # preventing the discontinuity at the foveal transition to parafovea.
    F = 30 * (20 / Fv) 

    deg = np.arange(0, Deg_max, 0.01)
    
    ang_res = []
    for d in deg:
        if d < e_0:
            if fov_lin:
                ang_res.append(angular_resolution)
            else:
                a = B ** (S / F)
                b = S / angular_resolution
                ang_res.append(S / (S / angular_resolution + (np.log2(d + a) / np.log2(B)) - b))
        else:
            ang_res.append(S / (d - e_0 + S / angular_resolution))
    
    ang = np.arctan(interpupildist_inch / distance) * (180 / np.pi)
    ang_idx = np.abs(deg - ang).argmin()
    eff_res = interpupildist_inch / (distance * np.tan((np.pi / 180) * (1 / (2 * ang_res[ang_idx]))))
    scale = eff_res / interpupildist_px
    
    
    return eff_res, scale

interpupildist_inch = 1  # in terms of inch
interpupildist_px = 187  # in terms of pixels

# Define the range of actual distances in feet
min_distance = 1
max_distance = 120
num_distances = 10  # Number of distances to generate logarithmically

# Initialize lists to store the results
eff_resolutions = []
scales = []

distances_feet = np.logspace(np.log10(min_distance), np.log10(max_distance), num=num_distances)


# Iterate over the range of distances
for actual_distance in distances_feet:
    simulated_distance = actual_distance / 3.5

    # Calculate eff_res_output and scale_output
    eff_res_output, scale_output = get_eff_resolution(interpupildist_inch, interpupildist_px, simulated_distance)

    # Append the results to the lists
    eff_resolutions.append(eff_res_output)
    scales.append(scale_output)
    
scale_factors = np.array(scales)
# Define the range of scale factors
# scale_factors = np.linspace(0.1, 1.0, num=10)  # Range from 0.1 to 1 with 10 values
# scale_factors = np.logspace(np.log10(0.1), np.log10(0.99), num=10)

#print("iteration:",_)

# Reverse the order of scale factors
scale_factors = scale_factors[::-1]

# Initialize lists to store accuracy values for each run
accuracy_values_runs = []

# Number of independent runs
num_runs = 10

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Concatenate X_train and X_test, y_train and y_test
X_full = np.concatenate((X_train, X_test))
y_full = np.concatenate((y_train, y_test))

# Iterate over each run
for _ in range(num_runs):
    print("iteration:",_)
    accuracy_values = []  # Initialize accuracy values for this run
    
    # Shuffle the data
    X_full_shuffled, y_full_shuffled = shuffle(X_full, y_full, random_state=_*42)

    # Repartition the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_full_shuffled, y_full_shuffled, test_size=0.2, random_state=_*42)

    # Preprocess the data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=0)
    
    # Iterate over each scale factor
    for scale_factor in scale_factors:
        # Calculate the target size based on the scale factor
        target_size = (int(X_test.shape[1] * scale_factor), int(X_test.shape[2] * scale_factor))
        
        # Resize all images in the CIFAR10 test dataset by the scale factor
        downscaled_X_test = tf.image.resize(X_test, size=target_size)
        upscaled_X_test = tf.image.resize(downscaled_X_test, size=(X_test.shape[1], X_test.shape[2]))

        # Evaluate the model on scaled test images
        loss, score = model.evaluate(upscaled_X_test, y_test, verbose=0)
        
        # Append the accuracy value to the list
        accuracy_values.append(score)
    
    # Append accuracy values of this run to the list
    accuracy_values_runs.append(accuracy_values)

# Convert accuracy values to numpy array for easier manipulation
accuracy_values_runs = np.array(accuracy_values_runs)

# Calculate mean and standard deviation for each scale parameter
mean_accuracy = np.mean(accuracy_values_runs, axis=0)
std_accuracy = np.std(accuracy_values_runs, axis=0)

# plot results
plt.plot(distances_feet, mean_accuracy, marker='o', label='CNN')
plt.fill_between(distances_feet, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy, alpha=0.3)
plt.xlabel('Physical Distance (feet)')
plt.ylabel('Top-1 Accuracy')
plt.title('Performance of a CNN (w Human frontend) as a function of Physical Distance', fontsize=9,color='red')
plt.grid(True, linestyle='--', linewidth=0.8, color='gray')
plt.xlim(0, 120)
plt.legend()
plt.savefig('PerfCNNwHumanfrontEnd.png')  # You can specify the filename and file format (e.g., 'figure.png', 'figure.pdf', 'figure.jpg')
plt.show()
