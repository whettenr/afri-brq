import os
from sklearn.cluster import MiniBatchKMeans, KMeans
import joblib

# function to do kmeans
def perform_kmeans(d, k=100, model_save_path="kmeans_model.joblib"):
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
    return preditions


# load all files from folder 
def load_and_concatenate_npy(folder_path):
    all_arrays = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = np.load(file_path)
                all_arrays.append(data)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    if not all_arrays:
        raise ValueError("No .npy files found or all failed to load.")
    d = np.concatenate(all_arrays, axis=0)
    print(d.shape)
    return d

def predict_with_saved_model(data_folder, model_path="kmeans_model.joblib"):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    new_data = load_and_concatenate_npy(data_folder)
    predictions = model.predict(new_data)
    return predictions

save_embed_folder = "/users/rwhetten/african_brq/quantization_framework/embeddings"
data = load_and_concatenate_npy(save_embed_folder)
labels = perform_kmeans(data, k=100)

new_folder = "/path/to/new/data"
new_labels = predict_with_saved_model(new_folder, model_path="kmeans_model.joblib")
# and conatenate to train 