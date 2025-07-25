# Complete training debugging script
import os
import sys
import tensorflow as tf
import keras
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modelling.model import build_fused_model 
from data.dataset import Dataset  

def debug_training_pipeline(config):
    """Complete debugging of your training pipeline."""
    
    print("=== TRAINING PIPELINE DEBUGGING ===")
    
    # 1. Load and inspect your processed data
    print("\n1. CHECKING SAVED DATA:")
    X = np.load(config.processed_dataset_dir + "train/X.npy")
    y = np.load(config.processed_dataset_dir + "train/y.npy")
    
    print(f"Saved X shape: {X.shape}")  # Should be (num_samples, window_size, num_features)
    print(f"Saved y shape: {y.shape}")  # Should be (num_samples,)
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")
    print(f"X range: [{X.min():.6f}, {X.max():.6f}]")
    print(f"y range: [{y.min()}, {y.max()}]")
    print(f"y unique values: {np.unique(y)}")
    
    # Check for problematic values in X
    print(f"X contains NaN: {np.isnan(X).any()}")
    print(f"X contains Inf: {np.isinf(X).any()}")
    print(f"X contains very large values (>1000): {(np.abs(X) > 1000).any()}")
    
    # 2. Create dataset and check batch
    print("\n2. CHECKING TENSORFLOW DATASET:")
    dataset = Dataset(config.random_seed, target='true_room')
    train_dataset = dataset.create_tf_dataset(
        config.processed_dataset_dir + "train/", 
        batch_size=32, 
        shuffle=True
    )
    
    # Get one batch for inspection
    for batch_x, batch_y, batch_seq in train_dataset.take(1):
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Batch y shape: {batch_y.shape}")
        print(f"Batch X dtype: {batch_x.dtype}")
        print(f"Batch y dtype: {batch_y.dtype}")
        print(f"Batch X range: [{tf.reduce_min(batch_x):.6f}, {tf.reduce_max(batch_x):.6f}]")
        print(f"Batch y values: {batch_y.numpy()}")
        
        # Check for problematic values
        print(f"Batch X contains NaN: {tf.reduce_any(tf.math.is_nan(batch_x))}")
        print(f"Batch X contains Inf: {tf.reduce_any(tf.math.is_inf(batch_x))}")
        
        sample_batch_x = batch_x
        sample_batch_y = batch_y
        break
    
    # 3. Create and test model
    print("\n3. CHECKING MODEL:")
    
    # Create hyperparameters (adjust these to match your actual hyperparameters)
    from keras_tuner import HyperParameters
    hp = HyperParameters()
    
    # Set your actual hyperparameters here
    hp.Fixed("xception_bool", True)
    hp.Fixed("num_filters", 32)
    hp.Fixed("kernel_size", 3)
    hp.Fixed("middle_blocks", 2)
    hp.Fixed("downsample", False)
    hp.Fixed("cbam_bool", True)
    hp.Fixed("r_ratio", 8)
    hp.Fixed("eca_bool", True)
    hp.Fixed("gamma", 2)
    hp.Fixed("beta", 1)
    hp.Fixed("gru_units", 64)
    hp.Fixed("fc_units", 64)
    hp.Fixed("learning_rate", 0.001)
    hp.Fixed("xception_dropout", 0.1)
    hp.Fixed("gru_dropout", 0.1)
    hp.Fixed("fc_dropout", 0.1)
    
    # Build model
    model = build_fused_model(hp)

    # Build model with actual data shape
    model.build(input_shape=(None, sample_batch_x.shape[1], sample_batch_x.shape[2]))
    print(f"Model built successfully with input shape: {sample_batch_x.shape}")
    model.summary()
    
    # 4. Test model prediction
    print("\n4. TESTING MODEL PREDICTION:")
    try:
        predictions = model(sample_batch_x, training=False)
        print(f"✅ Model prediction successful!")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction range: [{tf.reduce_min(predictions):.6f}, {tf.reduce_max(predictions):.6f}]")
        print(f"Sample prediction (first sample): {predictions[0].numpy()}")
        
        # Check if predictions sum to 1 (softmax property)
        pred_sums = tf.reduce_sum(predictions, axis=1)
        print(f"Prediction sums (should be ~1.0): {pred_sums[:3].numpy()}")
        
        # Check for problematic predictions
        print(f"Predictions contain NaN: {tf.reduce_any(tf.math.is_nan(predictions))}")
        print(f"Predictions contain Inf: {tf.reduce_any(tf.math.is_inf(predictions))}")
        
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
        return False
    
    # 5. Test loss calculation
    print("\n5. TESTING LOSS CALCULATION:")
    try:
        # Test the loss function manually
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        loss_value = loss_fn(sample_batch_y, predictions)
        print(f"Manual loss calculation: {loss_value.numpy():.6f}")
        
        # Test individual sample losses
        sample_losses = keras.losses.sparse_categorical_crossentropy(sample_batch_y, predictions)
        print(f"Individual sample losses (first 5): {sample_losses[:5].numpy()}")
        print(f"Max individual loss: {tf.reduce_max(sample_losses).numpy():.6f}")
        print(f"Mean individual loss: {tf.reduce_mean(sample_losses).numpy():.6f}")
        
        # Check for extremely high losses
        high_loss_samples = tf.where(sample_losses > 10.0)
        if len(high_loss_samples) > 0:
            print(f"⚠️  Samples with loss > 10: {len(high_loss_samples.numpy())} samples")
            for i in range(min(3, len(high_loss_samples))):
                idx = high_loss_samples[i][0].numpy()
                print(f"  Sample {idx}: y={sample_batch_y[idx].numpy()}, pred={predictions[idx].numpy()}, loss={sample_losses[idx].numpy():.2f}")
        
        if loss_value.numpy() > 1000:
            print("❌ EXTREMELY HIGH LOSS DETECTED!")
            print("This suggests a serious issue with model or data.")
            return False
        else:
            print("✅ Loss calculation looks reasonable")
            
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        return False

    # 6. Test one training step
    print("\n6. TESTING TRAINING STEP:")
    try:
        # Test a single training step
        with tf.GradientTape() as tape:
            predictions = model(sample_batch_x, training=True)
            loss = model.compiled_loss(sample_batch_y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        print(f"Training step loss: {loss.numpy():.6f}")
        
        # Check gradient norms
        grad_norms = [tf.norm(g).numpy() if g is not None else 0.0 for g in gradients]
        max_grad_norm = max(grad_norms) if grad_norms else 0.0
        mean_grad_norm = np.mean([g for g in grad_norms if g > 0])
        
        print(f"Max gradient norm: {max_grad_norm:.6f}")
        print(f"Mean gradient norm: {mean_grad_norm:.6f}")
        
        if max_grad_norm > 100:
            print("⚠️  Very high gradients detected - possible exploding gradients")
        elif max_grad_norm < 1e-7:
            print("⚠️  Very small gradients detected - possible vanishing gradients")
        else:
            print("✅ Gradient norms look reasonable")
            
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False
    
    print("\n=== DEBUGGING COMPLETE ===")
    return True

# Usage:
if __name__ == "__main__":
    from config import Config  # Import your config
    config = Config()
    
    success = debug_training_pipeline(config)
    
    if success:
        print("\n✅ All checks passed! Your model should train normally.")
    else:
        print("\n❌ Issues detected. Fix the problems above before training.")