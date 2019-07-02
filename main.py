from kmeans_parallel import main as kmeans
from GA_parallel import parallel_GA

import argparse
import pickle
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('--uav', type=int, default=6,
                    help='number of UAVs (default: 6)')
parser.add_argument('--threshold', type=int, default=100000,
                    help='population threshold for filtering cities'
                         +' (default: 100000)')
args = parser.parse_args()

print('Problem Settings:')
print('Number of UAVs:', args.uav)
print('Consider cities with population more than', args.threshold)
print()
os.makedirs('output/', exist_ok=True)



# Filter data and run K-means
clustered_path = f'data/kmeans_result_{args.threshold}_{args.uav}.pkl'
if os.path.exists(clustered_path):
    print('Found clustered result. Loading...')
    with open(clustered_path, 'rb') as f:
        clustered = pickle.load(f)
else:
    print('Running K-means...')
    out = kmeans('data/uscitiesv1.4.csv', args.uav, PL=args.threshold,
                 plot_clustered=True, plot_unclustered=False,
                 plot_init_centroid=False)
    centroid_x, centroid_y, labels, x, y, runtime = out
    clustered = []
    for cluster in np.unique(labels):
        idx = labels == cluster
        lng = x[idx]
        lat = y[idx]
        df = pd.DataFrame({'lng': lng, 'lat': lat})
        clustered.append(df)
    with open(clustered_path, 'wb') as f:
        pickle.dump(clustered, f)



# Solve with Genetic Algorithm
print()
print('Solving with the Genetic Algorithm...')
for i, cluster in enumerate(clustered, 1):
    cluster = cluster[['lng','lat']].astype(np.float32)
    all_pop, all_fitness = parallel_GA(cluster)
    best = all_pop[-1][np.argmin(all_fitness[-1]),:]
    cluster = cluster.iloc[best].to_numpy()
    cluster = np.concatenate((cluster, cluster[[0],:]))
    plt.plot(cluster[:,0], cluster[:,1], '-o', markersize=3, label=f'UAV {i}')
    print('UAV', i, 'planned')
print('FINISH')
plt.title('Planned Paths')
plt.xlabel('longitude')
plt.ylabel('latitude')
#plt.legend()
plt.savefig('output/planned_paths.png')
