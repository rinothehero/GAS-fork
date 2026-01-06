import sys
import os
import numpy as np

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

from dataset import Shard_Dirichlet

# Create simple test data
np.random.seed(2023)
n_samples_per_class = 500
K = 10  # number of classes
y_train = np.concatenate([np.full(n_samples_per_class, i) for i in range(K)])

# Test parameters
n_parties = 20
shard = 2
alpha = 0.1

print("=" * 80)
print("Testing Shard + Dirichlet Distribution")
print("=" * 80)
print(f"Total samples: {len(y_train)}")
print(f"Number of classes: {K}")
print(f"Number of clients: {n_parties}")
print(f"Classes per client (shard): {shard}")
print(f"Dirichlet alpha: {alpha}")
print("=" * 80)

# Run the Shard_Dirichlet function
party2dataidx = Shard_Dirichlet(y_train, n_parties, K=K, alpha=alpha, shard=shard, min_require_size=10)

print("\n" + "=" * 80)
print("Summary Statistics")
print("=" * 80)

# Verify each client has exactly 'shard' classes
for i in range(n_parties):
    labels = np.unique(y_train[party2dataidx[i]])
    print(f"Client {i}: {len(party2dataidx[i])} samples, "
          f"{len(labels)} classes (expected {shard}), "
          f"classes: {sorted(labels.tolist())}")

    # Check if exactly shard classes
    if len(labels) != shard:
        print(f"  ⚠️  WARNING: Expected {shard} classes but got {len(labels)}")

print("\n" + "=" * 80)
print("✅ Test completed!")
print("=" * 80)
