import os
from sklearn.cluster import MiniBatchKMeans, KMeans
import joblib
import numpy as np
from tqdm import tqdm
import random

# function to do kmeans
def perform_kmeans(d, k=500, model_save_path="kmeans_model.joblib"):
    k = int(k)
    mb_kmeans = MiniBatchKMeans(n_clusters=k,
                                random_state=42,
                                batch_size=64 * k,
                                init='k-means++', n_init=1, init_size=100000,
                                reassignment_ratio=5e-4,
                                verbose=2)
    mb_kmeans.fit(d)
    predictions = mb_kmeans.labels_
    joblib.dump(mb_kmeans, model_save_path)
    print(f"Model saved to {model_save_path}")
    return predictions


# # load all files from folder 
# def load_and_concatenate_npy(folder_path):
#     all_arrays = []
#     for file_name in tqdm(os.listdir(folder_path)):
#         if file_name.endswith(".npy"):
#             file_path = os.path.join(folder_path, file_name)
#             try:
#                 data = np.load(file_path)
#                 all_arrays.append(data)
#             except Exception as e:
#                 print(f"Error loading {file_name}: {e}")
#     if not all_arrays:
#         raise ValueError("No .npy files found or all failed to load.")
#     d = np.concatenate(all_arrays, axis=0)
#     return d

def load_and_concatenate_npy(folder_path, sample_fraction=1.0, seed=42):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    if not all_files:
        raise ValueError("No .npy files found in the folder.")
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Sample the files
    sample_size = int(len(all_files) * sample_fraction)
    sampled_files = random.sample(all_files, sample_size)

    all_arrays = []
    for file_name in tqdm(sampled_files):
        file_path = os.path.join(folder_path, file_name)
        try:
            data = np.load(file_path)
            all_arrays.append(data)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    
    if not all_arrays:
        raise ValueError("No valid .npy files were loaded.")
    
    return np.concatenate(all_arrays, axis=0)

def predict_with_saved_model(folder_path, model_path="kmeans_model.joblib", save_loc=None):

    all_predictions = []
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = np.load(file_path)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

            predictions = model.predict(data)
            if save_loc:
                np.save(f"{save_loc}/{file_name}", predictions.astype(np.uint16))

save_embed_folder = "/users/rwhetten/african_brq/quantization_framework/embeddings/CappFM2"
data = load_and_concatenate_npy(save_embed_folder, sample_fraction=0.25)
labels = perform_kmeans(data, k=500)


new_folder = save_embed_folder
train_save_loc = "/users/rwhetten/african_brq/quantization_framework/targets/kmean_train"

predict_with_saved_model(save_embed_folder, model_path="kmeans_model.joblib", save_loc=train_save_loc)


new_folder = "/users/rwhetten/african_brq/quantization_framework/embeddings/valid"
valid_save_loc = "/users/rwhetten/african_brq/quantization_framework/targets/kmean_valid"

predict_with_saved_model(new_folder, model_path="kmeans_model.joblib", save_loc=valid_save_loc)

# and conatenate to train 