import numpy as np
from scipy.ndimage import label, gaussian_filter, convolve
from skimage.graph import route_through_array
from collections import deque


# 8-connected neighbor offsets
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

def build_adjacency(edge_map):
    """Build adjacency dictionary for each edge pixel in the edge map. 
    Each key is a pixel coordinate (y, x) and the value is a list of neighboring coordinates.
    """
    edge_map = edge_map.astype(np.uint8)
    coords = [tuple(c) for c in np.argwhere(edge_map > 0)] # We are interested in the edge pixels
    coord_set = set(coords)
    adj = {}
    for y, x in coords:
        neighbors = [(y+dy, x+dx) for dy, dx in NEIGHBOR_OFFSETS
                     if (y+dy, x+dx) in coord_set]
        adj[(y, x)] = neighbors
    return adj

def find_endpoints(adj):
    """Return list of endpoints. These are edge pixels with exactly 1 neighbor."""
    return [p for p, nbrs in adj.items() if len(nbrs) == 1]

def connect_components_shortest_path(binary_map):
    """
    Connect disconnected components in a binary map using the shortest path.

    Parameters:
        binary_map (numpy.ndarray): 2D array of 0s and 1s.

    Returns:
        numpy.ndarray: Updated binary map with all components connected.
    """
    # Ensure input is binary
    binary_map = (binary_map > 0).astype(np.uint8)

    # Label connected components
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connected neighbourhood
    labeled, num_features = label(binary_map, structure=structure)
    if num_features <= 1:
        return binary_map  # Already connected

    # We'll iteratively connect components until only one remains
    while num_features > 1:
        # Get coordinates of the first component
        coords1 = np.argwhere(labeled == 1)
        
        min_dist = np.inf
        best_pair = None
        
        # Compare to all other components
        for comp_id in range(2, num_features + 1):
            coords2 = np.argwhere(labeled == comp_id)

            # Compute pairwise distances (brute force)
            for p1 in coords1:
                for p2 in coords2:
                    dist = np.sum((p1 - p2) ** 2)  # squared distance
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (tuple(p1), tuple(p2))
        
        # Create a cost array where 0s have low cost and 1s have high cost
        # This ensures path goes mostly through 0s
        cost_array = np.where(binary_map == 0, 1, 10)

        # Find shortest path between the closest pixels
        start, end = best_pair
        path, _ = route_through_array(cost_array, start, end, fully_connected=True)
        path = np.array(path)

        # Set path pixels to 1
        binary_map[path[:, 0], path[:, 1]] = 1

        # Recompute connected components
        labeled, num_features = label(binary_map, structure=structure)

    return binary_map

def bfs_longest_path(adj, start):
    """Breadth-first search to find the longest path from a starting point."""

    visited = {start: None} # Dictionary to track visited nodes and their parents
    q = deque([start]) # Initialize the queue with the starting node
    while q:
        node = q.popleft() # Get the next node from the queue
        for nbr in adj[node]:  # Iterate over neighbors of the node 
            if nbr not in visited: 
                visited[nbr] = node 
                q.append(nbr)

    # Farthest node found
    farthest = node # the farthest node will be the last node processed

    # Reconstruct path. This will be the longest path from the start node. 
    path = []
    while farthest is not None:
        path.append(farthest)
        farthest = visited[farthest]
    return path[::-1]  # Return the path in the correct order (from start to farthest node)

def clean_edge_skeleton(skel):
    """
    Remove tangents/junctions from skeleton to produce:
    - exactly 2 endpoints
    - all other points have degree 2
    Returns:
        cleaned_skel (binary array)
        ordered_coords (list of (y, x) pixels from endpoint to endpoint)
    """
    adj = build_adjacency(skel)
    endpoints = find_endpoints(adj)
    if len(endpoints) < 2:
        raise ValueError(f"Expected at least 2 endpoints, found {len(endpoints)}")

    # Find the longest path from any endpoint. The longest line should connect the real endpoints.
    max_path = []
    for ep in endpoints:
        path = bfs_longest_path(adj, ep)
        if len(path) > len(max_path):
            max_path = path

    # Keep only pixels in longest path
    cleaned = np.zeros_like(skel, dtype=np.uint8)
    for y, x in max_path:
        cleaned[y, x] = 1

    return cleaned, max_path

def clean_probabilistic_edge(edge_map, threshold=0.1,sigma=5):
    """Cleans a probabilistic edge map so it is suitable for skeletonization and line extraction."""
    edge = edge_map.copy()

    # smooth guassin filter
    edge = gaussian_filter(edge, sigma=sigma)

    # Inital thrshold to remove low prob pixels
    edge = np.where(edge >= threshold, edge, 0)

    # round the edge map to reduce number of unique values
    edge = np.round(edge, 2)

    #Check that cleaned edge map is one connected object
    binary_mask = (edge > 0).astype(np.uint8)

    #count connected components
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connected neighbourhood
    labeled, num_features = label(binary_mask, structure=structure)
    
    assert num_features == 1, f"Cleaned edge map has {num_features} connected components, expected 1."


    return edge, binary_mask

def skeletonize_zhang_suen(binary_image, prob_map=None, max_iterations=10_000, return_iterations=False):
    """
    Vectorized (fast) Zhang–Suen skeletonization.
    If prob_map is provided, deletion is probability-guided (low prob removed first).
    
    Parameters
    ----------
    binary_image : 2D np.ndarray of 0s and 1s
    prob_map : 2D np.ndarray of floats in [0,1] or None
    max_iterations : int
        Maximum iterations to prevent infinite loops.
        
    Returns
    -------
    skeleton : 2D np.ndarray of 0s and 1s
    """
    # Pad input to handle borders easily
    A = np.pad(binary_image.astype(np.uint8), 1)
    if prob_map is not None:
        P = np.pad(prob_map.astype(float), 1)
    
    # Neighbor counting kernel
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    
    iteration = 0
    changed = True
    iterations = [] # The edge map at each iteration

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        for phase in [1, 2]:
            # --- Compute neighbor count for all pixels ---
            N = convolve(A, kernel, mode='constant', cval=0)
            
            # --- Compute transitions (0→1) in 8-neighborhood sequence ---
            P2 = np.roll(A, -1, axis=0)
            P3 = np.roll(np.roll(A, -1, axis=0), -1, axis=1)
            P4 = np.roll(A, -1, axis=1)
            P5 = np.roll(np.roll(A, 1, axis=0), -1, axis=1)
            P6 = np.roll(A, 1, axis=0)
            P7 = np.roll(np.roll(A, 1, axis=0), 1, axis=1)
            P8 = np.roll(A, 1, axis=1)
            P9 = np.roll(np.roll(A, -1, axis=0), 1, axis=1)

            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            transitions = np.zeros_like(A)
            for k in range(8):
                transitions += ((seq[k] == 0) & (seq[k+1] == 1))

            # --- Deletion mask: Zhang–Suen rules ---
            condA = (A == 1)
            condB = (N >= 2) & (N <= 6)
            condC = (transitions == 1)

            if phase == 1:
                condD = (P2 * P4 * P6 == 0)
                condE = (P4 * P6 * P8 == 0)
            else:
                condD = (P2 * P4 * P8 == 0)
                condE = (P2 * P6 * P8 == 0)

            deletable = condA & condB & condC & condD & condE
            
            # --- If probability map is provided: remove only low-confidence pixels ---
            if prob_map is not None:
                if np.any(deletable):
                    # keep only those with lowest probability
                    min_prob = P[deletable].min()
                    mask = deletable & (np.isclose(P, min_prob))
                    A[mask] = 0
                    changed = True
            else:
                # Original Zhang–Suen deletion
                if np.any(deletable):
                    A[deletable] = 0
                    changed = True

            deleted_this_iter = np.sum(deletable)
            if prob_map is None:
                print(f"Iteration {iteration}, Phase {phase}, Pixels deleted: {deleted_this_iter}")
            else:
                
                print(f"Iteration {iteration}, Phase {phase}, Pixels deleted: {deleted_this_iter}, Min prob: {min_prob}")

        if return_iterations:
            iterations.append(A[1:-1, 1:-1].copy())
             
    if return_iterations:
        return A[1:-1, 1:-1], iterations
    
    return A[1:-1, 1:-1]
