import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# create_projection_head and get_prototypes functions remain the same.
def create_projection_head(input_shape, num_layers=3, initial_neurons=128):
    """Creates a multi-layer projection head for metric learning."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    neurons = initial_neurons
    for _ in range(num_layers):
        x = layers.Dense(neurons, activation='relu')(x)
        neurons //= 2
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_prototypes(X, y, n_prototypes_per_class=5):
    """Selects prototypes using K-Means clustering for each class."""
    prototypes = []
    labels = []
    for class_id in np.unique(y):
        class_samples = X[y == class_id]
        n_clusters = min(n_prototypes_per_class, len(class_samples))
        if n_clusters == 0:
            continue
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(class_samples)
        prototypes.extend(kmeans.cluster_centers_)
        labels.extend([class_id] * n_clusters)
    return np.array(prototypes), np.array(labels)

# ======================= START OF CORRECTED CODE =======================

def contrastive_loss(dissimilarity_matrix, y_batch, temperature=0.5):
    """
    Computes the contrastive loss from a full dissimilarity matrix.
    dissimilarity_matrix: A [2*batch_size, 2*batch_size] matrix.
    y_batch: The labels for the batch of size 2*batch_size.
    """
    # The paper's NT-Xent loss aims to minimize dissimilarity for positive pairs
    # and maximize it for negative pairs.
    # In NT-Xent, similarity is used. We can define similarity = 1 - dissimilarity.
    similarity_matrix = 1.0 - dissimilarity_matrix

    # Create a mask to identify positive pairs (samples with the same label)
    labels = tf.expand_dims(y_batch, 1)
    positive_mask = tf.equal(labels, tf.transpose(labels))
    # Remove self-comparisons
    identity_mask = tf.eye(tf.shape(y_batch)[0], dtype=tf.bool)
    positive_mask = positive_mask & ~identity_mask

    # Negative pairs are all pairs that are not positive and not self-comparisons
    negative_mask = ~positive_mask & ~identity_mask
    
    # Use masks to select the similarities for positive and negative pairs
    positive_similarities = tf.boolean_mask(similarity_matrix, positive_mask)
    negative_similarities = tf.boolean_mask(similarity_matrix, negative_mask)

    # Calculate the NT-Xent loss
    # The loss for each sample is -log( sum(exp(sim_pos/T)) / sum(exp(sim_neg/T)) )
    # A simplified but effective version is to pull positives together and push negatives apart.
    exp_pos = tf.exp(positive_similarities / temperature)
    exp_neg = tf.exp(negative_similarities / temperature)
    
    numerator = tf.reduce_sum(exp_pos)
    denominator = numerator + tf.reduce_sum(exp_neg)
    
    loss = -tf.math.log(numerator / denominator)
    return loss

# In contrastive_learning.py

def train_contrastive_model(X_train, y_train, epochs=300, batch_size=32, temperature=0.5):
    """Main function to train the contrastive dissimilarity model with a learning rate scheduler."""
    print("--- Starting Contrastive Dissimilarity Training ---")

    # The paper uses SMOTE only on imbalanced datasets. Since yours is balanced, we can skip it.
    X_resampled, y_resampled = X_train, y_train
    X_resampled = X_resampled.astype(np.float32)

    input_shape = X_train.shape[1:]
    projection_head = create_projection_head(input_shape)

    # --- START OF CHANGES ---

    # 1. Create a learning rate schedule
    # This will decrease the learning rate over time, which helps escape plateaus.
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,  # Start with the paper's initial rate
        decay_steps=1000,
        decay_rate=0.9)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # --- END OF CHANGES ---

    dataset = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled)).shuffle(1000).batch(batch_size)

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for i, (x_batch, y_batch) in enumerate(dataset):
            d_prime = x_batch
            d_double = x_batch
            
            x_combined = tf.concat([d_prime, d_double], axis=0)
            n_combined = tf.shape(x_combined)[0]
            
            x_tiled = tf.tile(tf.expand_dims(x_combined, 1), [1, n_combined, 1])
            x_repeated = tf.tile(tf.expand_dims(x_combined, 0), [n_combined, 1, 1])
            
            all_pairs_diff = tf.abs(x_tiled - x_repeated)

            with tf.GradientTape() as tape:
                reshaped_diffs = tf.reshape(all_pairs_diff, [-1, input_shape[0]])
                dissimilarities_flat = projection_head(reshaped_diffs, training=True)
                
                dissimilarity_matrix = tf.reshape(dissimilarities_flat, (n_combined, n_combined))
                
                y_combined = tf.concat([y_batch, y_batch], axis=0)
                
                # Pass the temperature parameter to the loss function
                loss = contrastive_loss(dissimilarity_matrix, y_combined, temperature=temperature)
            
            grads = tape.gradient(loss, projection_head.trainable_variables)
            optimizer.apply_gradients(zip(grads, projection_head.trainable_variables))
            total_loss += loss
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / (i+1):.4f}")
    
    print("--- Contrastive Training Finished ---")
    return projection_head

# ======================= END OF CORRECTED CODE =======================