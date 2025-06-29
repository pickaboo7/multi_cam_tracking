import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pandas as pd
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')


class MultivisionTracker:

    def __init__(self, base_path, num_views=4, config=None):
        self.base_path = base_path
        self.num_views = num_views
        self.view_data = {}
        self.filtered_positions = {}
        self.transformation_matrices = {}
        self.plot_bounds = {}
        self.clustered_positions = {}
        #self.frame_counter = 0
 
        # attributes for tracking
        self.next_merged_id = 200  # Start merged IDs from 200
        self.merged_id_pairs = {}  
        self.cluster_tracking_ids = {}  
        self.tracking_id_last_seen = {}  
        self.current_frame = 0  
        self.cluster_history = [] 
        
        # Kalman filter tracking
        self.cluster_kalman_filters = {}  
        self.next_cluster_id = 1000      
        self.cluster_state = {}           
        self.cluster_id_history = {}     
        self.last_seen_frame = {}         
        self.max_lost_frames = 30         
        
        # Configuration with defaults
        
        self.config = {
            # Point merging


            'point_merge_threshold': 10,  
            'frame_merge_threshold': 15,  
 
            # Static filtering
            'static_min_occurrences': 200,  
            'static_proximity_threshold': 2,  
 
            # Clustering
            'cluster_distance_threshold': 700, 
            'min_samples_for_cluster': 2,  
            'max_views_per_cluster': 4,  
            'max_clusters_per_frame': 3, 
 
            # View 2 specific
            'view2_reduced_weight': 0.5,  
            'view2_full_weight': 1.0,   
            'view2_x_bound': -2179,      
 
            # Tracking
            'tracking_memory_frames': 150, 
            'tracking_cleanup_interval': 50, 
 
            # Kalman filter tracking
            'max_lost_frames': 30,        
            'cluster_dt': 1.0,            
            'cluster_process_noise': 0.01, 
            'cluster_measurement_noise': 10.0, 
            'cluster_match_threshold': 200.0,  
            'smooth_trajectories': True,   
 
            # Visualization
            'debug_mode': True,  
            'debug_max_frames': 10,  
        }
 
        # Override defaults with provided config
        if config:
            self.config.update(config)
 
    def load_view_data(self):
        print("Loading data for all views...")
        csv_centroids = {}
        alt_csv_centroids = {}
 
        for view_id in range(self.num_views):
            csv_file = f'centroids_source_{view_id}.csv'
            alt_csv_file = f'alt_centroids_source_{view_id}.csv'
 
            print(f"Looking for View {view_id}:")
            print(f"  CSV: {csv_file}")
            print(f"  Alt CSV: {alt_csv_file}")
 
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    frames_grouped = df.groupby('frame')
                    max_frame = df['frame'].max()
                    centroids_list = [None] * (max_frame + 1)
 
                    for frame, group in frames_grouped:
                        if 'tracking_id' in group.columns:
                            points = list(zip(
                                group['centroid_x'].tolist(),
                                group['centroid_y'].tolist(),
                                group['tracking_id'].tolist()
                            ))
                        else:
                            points = list(zip(
                                group['centroid_x'].tolist(),
                                group['centroid_y'].tolist(),
                                range(1, len(group) + 1)
                            ))
 
                        merged_points = self._merge_close_points(points, proximity_threshold=self.config['point_merge_threshold'])
                        centroids_list[frame] = merged_points
 
                    csv_centroids[view_id] = centroids_list
                    print(f"Loaded primary CSV with {len(df)} entries across {len(frames_grouped)} frames")
                except Exception as e:
                    print(f"Error loading CSV: {e}")
            else:
                print(f"CSV file missing!")
 
            if os.path.exists(alt_csv_file):
                try:
                    df = pd.read_csv(alt_csv_file)
                    frames_grouped = df.groupby('frame')
                    max_frame = df['frame'].max()
                    centroids_list = [None] * (max_frame + 1)
 
                    for frame, group in frames_grouped:
                        if 'tracking_id' in group.columns:
                            points = list(zip(
                                group['centroid_x'].tolist(),
                                group['centroid_y'].tolist(),
                                group['tracking_id'].tolist()
                            ))
                        else:
                            points = list(zip(
                                group['centroid_x'].tolist(),
                                group['centroid_y'].tolist(),
                                range(100, 100 + len(group))
                            ))
 
                        merged_points = self._merge_close_points(points, proximity_threshold=self.config['point_merge_threshold'])
                        centroids_list[frame] = merged_points
 
                    alt_csv_centroids[view_id] = centroids_list
                    print(f"Loaded alternative CSV with {len(df)} entries across {len(frames_grouped)} frames")
                except Exception as e:
                    print(f"Error loading alternative CSV: {e}")
            else:
                print(f"Alternative CSV file missing!")
 
        for view_id in range(1, self.num_views + 1):
            corners_file = f'{self.base_path}/points_cam_{view_id}.npy'
            image_file = f'{os.path.dirname(self.base_path)}/frame_cam_{view_id}.jpg'
 
            print(f"Looking for View {view_id} calibration data:")
            print(f"  Corners: {corners_file}")
            print(f"  Image: {image_file}")
 
            if not os.path.exists(corners_file):
                print(f"Corners file missing!")
                continue
 
            if view_id-1 not in csv_centroids and view_id-1 not in alt_csv_centroids:
                print(f"No centroids data available!")
                continue
 
            if not os.path.exists(image_file):
                print(f"Image file missing (optional)")
                image_file = None
 
            try:
                corners = np.load(corners_file, allow_pickle=True)
                if len(corners.shape) == 3:
                    corners = corners[0]
 
                combined_centroids = []
                primary_centroids = csv_centroids.get(view_id-1, [])
                alt_centroids = alt_csv_centroids.get(view_id-1, [])
 
                if primary_centroids and alt_centroids:
                    max_frames = max(len(primary_centroids), len(alt_centroids))
 
                    for frame_idx in range(max_frames):
                        points1 = primary_centroids[frame_idx] if frame_idx < len(primary_centroids) else None
                        points2 = alt_centroids[frame_idx] if frame_idx < len(alt_centroids) else None
 
                        if points1 and points2:
                            merge_threshold = self.config['frame_merge_threshold']
                            merged_points = []
                            used2 = set()
 
                            for pt1 in points1:
                                found_match = False
                                for j, pt2 in enumerate(points2):
                                    if j in used2:
                                        continue
                                    dist = np.linalg.norm(np.array(pt1[:2]) - np.array(pt2[:2]))
                                    if dist <= merge_threshold:
                                        merged_point = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2, pt1[2])
                                        merged_points.append(merged_point)
                                        used2.add(j)
                                        found_match = True
                                        break
                                if not found_match:
                                    merged_points.append(pt1)
 
                            for j, pt2 in enumerate(points2):
                                if j not in used2:
                                    merged_points.append(pt2)
 
                            combined_centroids.append(merged_points if merged_points else None)
 
                        elif points1:
                            combined_centroids.append(points1)
                        elif points2:
                            combined_centroids.append(points2)
                        else:
                            combined_centroids.append(None)
                            
                elif primary_centroids:
                    combined_centroids = primary_centroids
                elif alt_centroids:
                    combined_centroids = alt_centroids
 
                self.view_data[view_id] = {
                    'corners': corners,
                    'centroids': combined_centroids,
                    'image_file': image_file
                }
 
                non_empty_frames = sum(1 for f in combined_centroids if f is not None and len(f) > 0)
                total_points = sum(len(f) for f in combined_centroids if f is not None)
 
                print(f"View {view_id}: {len(combined_centroids)} total frames, {non_empty_frames} non-empty frames")
                print(f"Combined {total_points} total points")
                print(f"Corners shape: {corners.shape}")
            except Exception as e:
                print(f"Error loading view {view_id}: {e}")
 
        print(f"\nSuccessfully loaded {len(self.view_data)} views")
    
 
    def init_kalman_filter(self):
        """Initialize a new Kalman filter for tracking a cluster in 2D space"""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
        dt = self.config['cluster_dt']
        
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement function (only position is measured)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(2) * self.config['cluster_measurement_noise']
        
        # Process noise
        q = self.config['cluster_process_noise']
        kf.Q = np.array([
            [q/4*dt**4, 0, q/2*dt**3, 0],
            [0, q/4*dt**4, 0, q/2*dt**3],
            [q/2*dt**3, 0, q*dt**2, 0],
            [0, q/2*dt**3, 0, q*dt**2]
        ])
        
        # Initial state uncertainty
        kf.P *= 100
        
        return kf
    
    def track_clusters(self):
        """Apply Kalman filtering to clusters after group_centroids_across_views runs"""
        print("Applying Kalman filtering to cluster tracking...")
        
        # Process frames in order
        for frame_idx in sorted(self.clustered_positions.keys()):
            clusters = self.clustered_positions[frame_idx]
            self.cluster_id_history[frame_idx] = {}
            
            # Predict step for all existing Kalman filters
            for cluster_id, kf in self.cluster_kalman_filters.items():
                kf.predict()
                
                # Store predicted state
                if cluster_id in self.cluster_state:
                    self.cluster_state[cluster_id]['predicted'] = np.array([kf.x[0], kf.x[1]])
            
            # Match current clusters with predicted positions
            matched_filters = set()
            matched_clusters = set()
            matches = []  # (cluster_idx, cluster_id)
            
            # First try to match by tracking IDs if available
            for i, cluster in enumerate(clusters):
                if 'tracking_ids' not in cluster:
                    continue
                    
                # Try to match with existing cluster by tracking IDs
                best_match = None
                best_overlap = 0
                
                for cluster_id, state in self.cluster_state.items():
                    if cluster_id in matched_filters:
                        continue
                        
                    if 'tracking_ids' not in state:
                        continue
                        
                    # Count how many tracking IDs match
                    overlap = len(set(cluster['tracking_ids']).intersection(state['tracking_ids']))
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = cluster_id
                
                # Match if at least one tracking ID overlaps
                if best_match and best_overlap > 0:
                    matches.append((i, best_match))
                    matched_filters.add(best_match)
                    matched_clusters.add(i)
                    
                    # Update tracking IDs in state
                    self.cluster_state[best_match]['tracking_ids'] = cluster['tracking_ids']
            
            # Next, match by position for remaining clusters
            for i, cluster in enumerate(clusters):
                if i in matched_clusters:
                    continue
                    
                best_match = None
                best_dist = float('inf')
                
                for cluster_id, kf in self.cluster_kalman_filters.items():
                    if cluster_id in matched_filters:
                        continue
                        
                    # Calculate distance between cluster and predicted position
                    predicted_pos = self.cluster_state[cluster_id]['predicted']
                    cluster_pos = cluster['centroid']
                    dist = np.linalg.norm(predicted_pos - cluster_pos)
                    
                    if dist < best_dist and dist < self.config['cluster_match_threshold']:
                        best_dist = dist
                        best_match = cluster_id
                
                if best_match:
                    matches.append((i, best_match))
                    matched_filters.add(best_match)
                    matched_clusters.add(i)
            
            # Update Kalman filters for matched clusters
            for cluster_idx, cluster_id in matches:
                cluster = clusters[cluster_idx]
                kf = self.cluster_kalman_filters[cluster_id]
                pos = cluster['centroid']
                
                # Update the Kalman filter with new measurement
                kf.update(pos)
                
                # Store updated state and last seen frame
                if self.config['smooth_trajectories']:
                    # Use the Kalman filter's smoothed position
                    smoothed_pos = np.array([kf.x[0], kf.x[1]])
                    original_pos = cluster['centroid'].copy()
                    
                    # Update cluster position while preserving original points
                    offset = smoothed_pos - original_pos
                    
                    # Store original position for reference
                    cluster['original_centroid'] = original_pos
                    cluster['centroid'] = smoothed_pos
                    
                    # Update points if using smoothing (maintains relative positions)
                    if 'points' in cluster:
                        cluster['points'] = [p + offset for p in cluster['points']]
                
                self.cluster_state[cluster_id] = {
                    'position': np.array([kf.x[0], kf.x[1]]),
                    'velocity': np.array([kf.x[2], kf.x[3]]),
                    'tracking_ids': cluster['tracking_ids'] if 'tracking_ids' in cluster else []
                }
                self.last_seen_frame[cluster_id] = frame_idx
                
                # Store cluster ID for this frame
                self.cluster_id_history[frame_idx][cluster_idx] = cluster_id
                
                # Add consistent cluster ID to the cluster for visualization
                cluster['cluster_id'] = cluster_id
            
            # Create new Kalman filters for unmatched clusters
            for i, cluster in enumerate(clusters):
                if i in matched_clusters:
                    continue
                    
                # Create new cluster ID and Kalman filter
                new_id = self.next_cluster_id
                self.next_cluster_id += 1
                
                kf = self.init_kalman_filter()
                pos = cluster['centroid']
                
                # Initialize with current position and zero velocity
                kf.x = np.array([pos[0], pos[1], 0, 0])
                kf.update(pos)
                
                self.cluster_kalman_filters[new_id] = kf
                self.cluster_state[new_id] = {
                    'position': np.array([kf.x[0], kf.x[1]]),
                    'velocity': np.array([kf.x[2], kf.x[3]]),
                    'tracking_ids': cluster['tracking_ids'] if 'tracking_ids' in cluster else []
                }
                self.last_seen_frame[new_id] = frame_idx
                
                # Store cluster ID for this frame
                self.cluster_id_history[frame_idx][i] = new_id
                
                # Add consistent cluster ID to the cluster for visualization
                cluster['cluster_id'] = new_id
            
            # Clean up lost filters
            lost_filters = []
            for cluster_id in self.cluster_kalman_filters:
                if cluster_id not in matched_filters:
                    frames_lost = frame_idx - self.last_seen_frame[cluster_id]
                    if frames_lost > self.config['max_lost_frames']:
                        lost_filters.append(cluster_id)
            
            for cluster_id in lost_filters:
                del self.cluster_kalman_filters[cluster_id]
                del self.cluster_state[cluster_id]
                del self.last_seen_frame[cluster_id]
        
        print(f"Assigned {self.next_cluster_id - 1000} stable cluster IDs")
        return self.next_cluster_id - 1000
    
    

    def _merge_close_points(self, points, proximity_threshold=10):
        #don't require it anymore but removing the function calls would be more work :/
        return points
 
    def filter_static_points(self, min_occurrences=None, proximity_threshold=None):
        if min_occurrences is None:
            min_occurrences = self.config['static_min_occurrences']
        if proximity_threshold is None:
            proximity_threshold = self.config['static_proximity_threshold']
 
        print(f"Filtering static points (min_occurrences={min_occurrences}, proximity_threshold={proximity_threshold})...")
 
        for view_id, data in self.view_data.items():
            centroids = data['centroids']
            filtered_centroids = []
            point_counts = defaultdict(int)
 
            for frame_idx, frame_data in enumerate(centroids):
                if frame_data is not None and len(frame_data) > 0:
                    for point in frame_data:
                        rounded_point = (
                            round(point[0] / proximity_threshold) * proximity_threshold,
                            round(point[1] / proximity_threshold) * proximity_threshold
                        )
                        point_counts[rounded_point] += 1
 
            static_points = {rp for rp, count in point_counts.items() if count >= min_occurrences}
            print(f"View {view_id}: Found {len(static_points)} static locations")
 
            for frame_idx, frame_data in enumerate(centroids):
                if frame_data is not None and len(frame_data) > 0:
                    filtered_frame = []
                    for point in frame_data:
                        rounded_point = (
                            round(point[0] / proximity_threshold) * proximity_threshold,
                            round(point[1] / proximity_threshold) * proximity_threshold
                        )
                        if rounded_point not in static_points:
                            filtered_frame.append(point)
                    filtered_centroids.append(filtered_frame if filtered_frame else None)
                else:
                    filtered_centroids.append(None)
 
            self.filtered_positions[view_id] = filtered_centroids
 
    def create_homography_matrices(self, output_size=(1800, 1200)):
        print("Creating homography matrices...")
        width, height = output_size
        table_width, table_height = 1000, 400
        center_x, center_y = width // 2, height // 2
 
        dst_points = np.array([
            [center_x - table_width//2, center_y - table_height//2],
            [center_x + table_width//2, center_y - table_height//2],
            [center_x + table_width//2, center_y + table_height//2],
            [center_x - table_width//2, center_y + table_height//2]
        ], dtype=np.float32)
 
        bounds = {'x_min': 0, 'x_max': width, 'y_min': 0, 'y_max': height,
                  'center_x': center_x, 'center_y': center_y}
 
        for view_id, data in self.view_data.items():
            corners = np.array(data['corners'], dtype=np.float32)
            if corners.shape == (4, 2):
                H = cv2.getPerspectiveTransform(corners, dst_points)
                self.transformation_matrices[view_id] = H
                self.plot_bounds[view_id] = bounds
                print(f"View {view_id} homography matrix created")
 
    def transform_points(self, points, H):
        """Transform list of points using homography H."""
        if len(points) == 0:
            return np.array([])
 
        # Make sure points are just x,y coordinates
        if isinstance(points[0], (tuple, list)) and len(points[0]) > 2:
            pts = np.array([(p[0], p[1]) for p in points], dtype=np.float32).reshape(-1, 1, 2)
        else:
            pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
 
        warped = cv2.perspectiveTransform(pts, H)
        return warped.reshape(-1, 2)
 
    def debug_transformed_points(self, frame_idx=0, max_debug_frames=None):
        if max_debug_frames is None:
            max_debug_frames = self.config['debug_max_frames']
 
        if frame_idx >= max_debug_frames:
            return
 
        print(f"\n=== DEBUGGING FRAME {frame_idx} ===")
        all_transformed = {}
 
        for view_id in sorted(self.view_data.keys()):
            if frame_idx < len(self.filtered_positions[view_id]):
                centroids = self.filtered_positions[view_id][frame_idx]
                if centroids is not None and len(centroids) > 0:
                    # Extract only x,y coordinates for transformation
                    points_xy = [(point[0], point[1]) for point in centroids]
                    transformed = self.transform_points(points_xy, self.transformation_matrices[view_id])
 
                    # Extract tracking IDs
                    tracking_ids = [point[2] for point in centroids]
 
                    # Apply View 2 filtering for debugging display
                    if view_id == 2:
                        original_count = len(transformed)
                        valid_mask = transformed[:, 0] >= self.config['view2_x_bound']
                        transformed = transformed[valid_mask]
                        tracking_ids = [tracking_ids[i] for i, valid in enumerate(valid_mask) if valid]
                        filtered_count = len(transformed)
 
                        if original_count != filtered_count:
                            print(f"View 2 DEBUG: filtered {original_count - filtered_count}/{original_count} points (x < {self.config['view2_x_bound']})")
 
                    if len(transformed) > 0:
                        all_transformed[view_id] = (transformed, tracking_ids)
                        print(f"View {view_id}: {len(centroids)} original -> {len(transformed)} transformed")
                        if len(transformed) > 0:
                            print(f"  Sample points: {transformed[:min(3, len(transformed))]} tracking_ids: {tracking_ids[:min(3, len(tracking_ids))]}")
                            if view_id == 2:
                                print(f"  View 2 x-range: {transformed[:, 0].min():.1f} to {transformed[:, 0].max():.1f}")
 
        # Calculate pairwise distances between views
        if len(all_transformed) >= 2:
            view_ids = list(all_transformed.keys())
            for i, view1 in enumerate(view_ids):
                for view2 in view_ids[i+1:]:
                    transformed1, tracking_ids1 = all_transformed[view1]
                    transformed2, tracking_ids2 = all_transformed[view2]
 
                    if len(transformed1) > 0 and len(transformed2) > 0:
                        distances = cdist(transformed1, transformed2)
                        min_dist = np.min(distances)
                        print(f"Min distance between View {view1} and View {view2}: {min_dist:.1f}")
                        if min_dist < 300:  # Show close pairs
                            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                            print(f"  Closest pair: View{view1}[{min_idx[0]}] (ID: {tracking_ids1[min_idx[0]]}) to View{view2}[{min_idx[1]}] (ID: {tracking_ids2[min_idx[1]]})")
        else:
            print("Not enough views with points for distance calculation")
 
    def find_closest_point_per_view(self, points, view_ids, tracking_ids, target_point):
        selected_points = []
        selected_views = []
        selected_tracking_ids = []
 
        unique_views = np.unique(view_ids)
        for view_id in unique_views:
            view_mask = view_ids == view_id
            view_points = points[view_mask]
            view_track_ids = [tracking_ids[i] for i, m in enumerate(view_mask) if m]
 
            if len(view_points) > 0:
                # Find closest point from this view to target
                distances = np.linalg.norm(view_points - target_point, axis=1)
                closest_idx = np.argmin(distances)
                selected_points.append(view_points[closest_idx])
                selected_views.append(view_id)
                selected_tracking_ids.append(view_track_ids[closest_idx])
 
        return np.array(selected_points), np.array(selected_views), np.array(selected_tracking_ids)
    
    
    
    def group_centroids_across_views(self, eps=None, min_samples=None, debug_mode=None):
        """
        Group centroids across views using spatial clustering first, then maintain 
        consistent number of clusters (max 3) by promoting individual points when needed.
        Promoted points appear identical to spatial clusters in visualization.
        """
        if eps is None:
            eps = self.config['cluster_distance_threshold']
        if min_samples is None:
            min_samples = self.config['min_samples_for_cluster']
        if debug_mode is None:
            debug_mode = self.config['debug_mode']

        print(f"Grouping centroids across views with distance threshold={eps}...")
        print(f"View 2 weightage: {self.config['view2_reduced_weight']} weight in 3-4 view groups, "
            f"{self.config['view2_full_weight']} weight in 2-view groups")
        
        min_frames = min(len(self.filtered_positions[view_id]) for view_id in self.filtered_positions)
        print(f"Processing {min_frames} frames of tracking data")
        
        # Initialize tracking structures
        self.clustered_positions = {}
        
        # Keep track of maximum clusters observed, but cap it at max_clusters_per_frame (3)
        max_clusters_observed = 0
        absolute_max_clusters = self.config['max_clusters_per_frame']
        
        # Queue to store points that were part of clusters (100 frames)
        cluster_points_history = []
        max_history_frames = 180
        print(f"Using history of {max_history_frames} frames for point promotion")
        
        # Track last known centroid positions of clusters
        last_cluster_centroids = []
        
        # For debug stats
        total_spatial_clusters = 0
        total_promoted_clusters = 0
        frames_with_promoted_clusters = 0
        
        for frame_idx in range(min_frames):
            self.current_frame = frame_idx
            
            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{min_frames} ({frame_idx/min_frames*100:.1f}%)")
            
            # Debug first few frames in detail
            if debug_mode and frame_idx < self.config['debug_max_frames']:
                self.debug_transformed_points(frame_idx)
                print(f"\nDetailed debug for frame {frame_idx}:")
            
            all_points = []
            view_ids = []
            tracking_ids = []
            
            # Collect all transformed points from all views for this frame
            for view_id in sorted(self.view_data.keys()):
                if frame_idx < len(self.filtered_positions[view_id]):
                    centroids = self.filtered_positions[view_id][frame_idx]
                    if centroids is not None and len(centroids) > 0:
                        # Extract tracking IDs
                        frame_tracking_ids = [point[2] for point in centroids]
                        
                        # Get only x,y coordinates for transformation
                        points_xy = [(point[0], point[1]) for point in centroids]
                        transformed_pts = self.transform_points(points_xy, self.transformation_matrices[view_id])
                        
                        # Apply global X-bound filtering for all views
                        valid_x_mask = (transformed_pts[:, 0] <= 4500) & (transformed_pts[:, 0] >= -1900)
                        
                        # ADDITIONAL VIEW 2 SPECIFIC FILTERING - REMOVE POINTS WITH X < 0  (this is done because cam 2 has too much skew error after transformation)
                        if view_id == 2:
                            original_length = len(transformed_pts)
                            view2_mask = transformed_pts[:, 0] >= 1300
                            valid_x_mask = valid_x_mask & view2_mask
                            
                            # Debug information about View 2 filtering
                            if debug_mode and frame_idx < self.config['debug_max_frames'] and np.any(~view2_mask):
                                removed_count = np.sum(~view2_mask)
                                print(f"  View 2: Removed {removed_count} points with x < 0")
                        
                        transformed_pts = transformed_pts[valid_x_mask]
                        filtered_tracking_ids = [frame_tracking_ids[i] for i, valid in enumerate(valid_x_mask) if valid]
                        
                        if len(transformed_pts) > 0:  # Only add if points remain after filtering
                            all_points.extend(transformed_pts)
                            view_ids.extend([view_id] * len(transformed_pts))
                            tracking_ids.extend(filtered_tracking_ids)
                            
                            if debug_mode and frame_idx < self.config['debug_max_frames']:
                                print(f"  View {view_id}: {len(transformed_pts)} points after filtering")



            
            if len(all_points) == 0:
                if debug_mode and frame_idx < self.config['debug_max_frames']:
                    print(f"  No points found in frame {frame_idx}, skipping")
                
                self.clustered_positions[frame_idx] = []
                continue
            
            all_points = np.array(all_points)
            view_ids = np.array(view_ids)
            tracking_ids = np.array(tracking_ids)
            
            if debug_mode and frame_idx < self.config['debug_max_frames']:
                print(f"  Total points across all views: {len(all_points)}")
                unique_views = np.unique(view_ids)
                print(f"  Points from views: {[int(v) for v in unique_views]}")
            
            if len(all_points) < 2:
                # Only one point, create a single cluster
                single_cluster = {
                    'centroid': all_points[0],
                    'views': [int(view_ids[0])],
                    'points': [all_points[0]],
                    'tracking_ids': [tracking_ids[0]]
                    # No 'type' field so it appears as a normal cluster
                }
                
                self.clustered_positions[frame_idx] = [single_cluster]
                
                # Update tracking ID last seen
                self.tracking_id_last_seen[tracking_ids[0]] = frame_idx
                
                if debug_mode and frame_idx < self.config['debug_max_frames']:
                    print(f"  Only one point found, creating single-point cluster")
                
                # Update max clusters (capped at absolute_max_clusters)
                max_clusters_observed = min(max_clusters_observed + 1 if max_clusters_observed < 1 else max_clusters_observed, absolute_max_clusters)
                continue
            
            # FIRST PRIORITY: Spatial clustering using greedy approach
            final_groups = []
            used_points = set()
            
            if debug_mode and frame_idx < self.config['debug_max_frames']:
                print(f"  Starting spatial clustering with distance threshold {eps}...")
            
            # Start with each point as a potential group center for remaining points
            for i, center_point in enumerate(all_points):
                if i in used_points:
                    continue
                    
                cluster_points = [center_point]
                cluster_indices = [i]
                cluster_views = [view_ids[i]]
                cluster_track_ids = [tracking_ids[i]]
                
                # Find points close to this center from different views
                for j, other_point in enumerate(all_points):
                    if j == i or j in used_points:
                        continue
                        
                    # Check if view is already in the cluster
                    if view_ids[j] in cluster_views:
                        continue
                        
                    # Calculate distance
                    distance = np.linalg.norm(center_point - other_point)
                    if distance <= eps:
                        cluster_points.append(other_point)
                        cluster_indices.append(j)
                        cluster_views.append(view_ids[j])
                        cluster_track_ids.append(tracking_ids[j])
                
                # Only create multi-view cluster if we have points from multiple views
                if len(np.unique(cluster_views)) >= min_samples:
                    # Calculate weighted centroid
                    if len(cluster_views) >= 3:
                        # Apply weights: view 2 gets reduced weight, others get full weight
                        weights = []
                        weighted_points = []
                        
                        for k, (point, view_id) in enumerate(zip(cluster_points, cluster_views)):
                            if view_id == 2:
                                weight = self.config['view2_reduced_weight']
                            else:
                                weight = self.config['view2_full_weight']
                            
                            weights.append(weight)
                            weighted_points.append(point * weight)
                        
                        # Calculate weighted centroid
                        weights = np.array(weights)
                        weighted_points = np.array(weighted_points)
                        weighted_centroid = np.sum(weighted_points, axis=0) / np.sum(weights)
                    else:
                        # For 2-view groups, calculate simple average
                        weighted_centroid = np.mean(cluster_points, axis=0)
                    
                    # Create multi-view cluster - NOTE: No 'type' field for consistent rendering
                    cluster = {
                        'centroid': weighted_centroid,
                        'views': [int(view) for view in np.unique(cluster_views)],
                        'points': cluster_points,
                        'tracking_ids': cluster_track_ids
                    }
                    
                    final_groups.append(cluster)
                    total_spatial_clusters += 1
                    
                    # Mark all points in this cluster as used
                    for idx in cluster_indices:
                        used_points.add(idx)
                        
                    # Update tracking IDs last seen
                    for track_id in cluster_track_ids:
                        self.tracking_id_last_seen[track_id] = frame_idx
                    
                    if debug_mode and frame_idx < self.config['debug_max_frames']:
                        view_list = [int(v) for v in np.unique(cluster_views)]
                        print(f"    Created spatial cluster from views {view_list} with {len(cluster_points)} points")
            
            # STEP 2: Update maximum observed clusters (capped at absolute_max_clusters)
            max_clusters_observed = min(max(len(final_groups), max_clusters_observed), absolute_max_clusters)
            
            if debug_mode and frame_idx < self.config['debug_max_frames']:
                print(f"  Current spatial clusters: {len(final_groups)}, max observed (capped): {max_clusters_observed}")
            
            # STEP 3: Store current cluster points in history for future reference
            current_cluster_track_points = {}
            for group in final_groups:
                for i, track_id in enumerate(group['tracking_ids']):
                    current_cluster_track_points[track_id] = group['points'][i]
            
            # Add to history, maintain limited size
            cluster_points_history.append(current_cluster_track_points)
            if len(cluster_points_history) > max_history_frames:
                cluster_points_history.pop(0)
            
            # STEP 4: Check if we need to promote individual points to reach max_clusters
            promoted_count = 0
            if len(final_groups) < max_clusters_observed:
                # Calculate how many additional clusters we need
                clusters_needed = max_clusters_observed - len(final_groups)
                
                if debug_mode and frame_idx < self.config['debug_max_frames']:
                    print(f"  Need {clusters_needed} more clusters to reach max of {max_clusters_observed}")
                
                # Find all candidate points (not yet used in clusters) that were previously in clusters
                candidates = []
                
                # Get all historical tracking IDs from our history
                historical_tracking_ids = set()
                for history_frame in cluster_points_history:
                    historical_tracking_ids.update(history_frame.keys())
                
                # Find points in current frame matching those tracking IDs
                for i, point in enumerate(all_points):
                    if i not in used_points and tracking_ids[i] in historical_tracking_ids:
                        candidates.append({
                            'index': i,
                            'point': point,
                            'view_id': int(view_ids[i]),
                            'tracking_id': tracking_ids[i]
                        })
                
                if debug_mode and frame_idx < self.config['debug_max_frames']:
                    print(f"  Found {len(candidates)} candidate points from tracking history")
                
                # If we have last known centroids, prioritize points closest to them
                if candidates and last_cluster_centroids:
                    for candidate in candidates:
                        # Find minimum distance to any last known centroid
                        min_dist = float('inf')
                        closest_centroid = None
                        
                        for centroid_info in last_cluster_centroids:
                            dist = np.linalg.norm(candidate['point'] - centroid_info['centroid'])
                            if dist < min_dist:
                                min_dist = dist
                                closest_centroid = centroid_info
                        
                        candidate['distance'] = min_dist
                        candidate['closest_centroid'] = closest_centroid
                    
                    # Sort by distance to nearest last centroid
                    candidates.sort(key=lambda c: c['distance'])
                    
                    if debug_mode and frame_idx < self.config['debug_max_frames'] and candidates:
                        print(f"  Sorted candidates by distance to last known centroids")
                        print(f"  Best candidate distance: {candidates[0]['distance']:.1f}")
                # Inside group_centroids_across_views method, update the promotion section:

                # Create clusters from top candidates (closest to previous centroids first)
                for i in range(min(clusters_needed, len(candidates))):
                    candidate = candidates[i]
                    
                    # Get the point and its view ID
                    point = candidate['point']
                    view_id = candidate['view_id']
                    track_id = candidate['tracking_id']
                    
                    # Important: Create EXACTLY the same structure as spatial clusters
                    # The animation code likely checks these specific fields
                    promoted_cluster = {
                        'centroid': point,            # The centroid position of the cluster
                        'views': [int(view_id)],      # List of view IDs (just one for promoted point)
                        'points': [point],            # Array of points in the cluster 
                        'tracking_ids': [track_id],   # Tracking IDs of points in the cluster
                        'is_promoted_cluster': True            #flag to differenciate from normal clusters
                    }
                    
                    # Add additional debug info to help diagnose visualization issues
                    if debug_mode and frame_idx < self.config['debug_max_frames']:
                        print(f"    PROMOTED POINT: Creating cluster with structure: {promoted_cluster}")
                    
                    final_groups.append(promoted_cluster)
                    promoted_count += 1
                    total_promoted_clusters += 1
                    used_points.add(candidate['index'])
                    
                    # Update tracking ID last seen
                    self.tracking_id_last_seen[track_id] = frame_idx


            
            # Check if we promoted any clusters
            if promoted_count > 0:
                frames_with_promoted_clusters += 1
            
            # Store current cluster centroids for next frame
            last_cluster_centroids = [
                {'centroid': group['centroid'], 'tracking_ids': group['tracking_ids']}
                for group in final_groups
            ]
            # Add remaining unused points as individual points (not clusters)
            # These won't be part of clustered_positions, so they'll appear as normal points
            
            
            
            
            # 2. Replace your current filtering code with this more aggressive version

            # First, sort ALL groups by number of views (descending) then by points (descending)
            sorted_groups = sorted(final_groups, key=lambda g: (-len(g['views']), -len(g['points'])))

            # STRICTLY enforce maximum clusters - take only top clusters after sorting
            final_filtered_clusters = sorted_groups[:absolute_max_clusters]

            # DEBUG: Check if we're cutting off any multi-view clusters
            if debug_mode and frame_idx < self.config['debug_max_frames']:
                multi_view_kept = sum(1 for g in final_filtered_clusters if len(g['views']) > 1)
                multi_view_total = sum(1 for g in sorted_groups if len(g['views']) > 1)
                
                if multi_view_total > absolute_max_clusters:
                    print(f"  WARNING: Frame {frame_idx} has {multi_view_total} multi-view clusters, keeping only {multi_view_kept}")
                
                # Show what we're keeping vs discarding
                print(f"  Keeping top {len(final_filtered_clusters)} clusters:")
                for i, grp in enumerate(final_filtered_clusters):
                    print(f"    Cluster {i+1}: {len(grp['views'])} views, {len(grp['points'])} points")

            # Store strictly filtered groups
            self.clustered_positions[frame_idx] = final_filtered_clusters

        
        # Print final statistics
        print("\nClustering completed:")
        print(f"Processed {min_frames} frames")
        print(f"Created {total_spatial_clusters} spatial clusters")
        print(f"Promoted {total_promoted_clusters} individual points to clusters in {frames_with_promoted_clusters} frames")
        print(f"Maximum clusters per frame observed: {max_clusters_observed} (capped at {absolute_max_clusters})")

        # Debug: analyze cluster results distribution
        cluster_counts = {}
        for frame_idx, clusters in self.clustered_positions.items():
            count = len(clusters)
            if count not in cluster_counts:
                cluster_counts[count] = 0
            cluster_counts[count] += 1
        
        print("\nCluster distribution:")
        for count in sorted(cluster_counts.keys()):
            frames = cluster_counts[count]
            percentage = frames / min_frames * 100
            print(f"  {count} clusters: {frames} frames ({percentage:.1f}%)")


    
    def _cleanup_inactive_tracking_ids(self, current_frame, inactive_threshold=10):
        """Clean up tracking IDs that haven't been seen in a while."""
        count_removed = 0
        ids_to_remove = []
        
        for tracking_id, last_seen_frame in self.tracking_id_last_seen.items():
            if current_frame - last_seen_frame > inactive_threshold:
                ids_to_remove.append(tracking_id)
        
        for tracking_id in ids_to_remove:
            del self.tracking_id_last_seen[tracking_id]
            count_removed += 1
        
        return count_removed  # Always return an integer

 
    def plot_debug_frame(self, frame_idx=0):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        bounds = next(iter(self.plot_bounds.values()))
        
        # [rest of your code remains the same]
        
        if frame_idx in self.clustered_positions:
            groups = self.clustered_positions[frame_idx]
            multi_groups = [g for g in groups if len(g['views']) > 1]
            single_groups = [g for g in groups if len(g['views']) == 1]
            
            for group in single_groups:
                view_id = group['views'][0]
                track_ids = group.get('tracking_ids', ['N/A'])
                color_idx = (view_id - 1) % len(colors)
                ax2.scatter(group['centroid'][0], group['centroid'][1],
                        c=colors[color_idx], s=50, alpha=0.5, marker='o')
                track_id_str = ",".join(map(str, track_ids))
                ax2.text(group['centroid'][0], group['centroid'][1], f"ID:{track_id_str}",
                    fontsize=8, ha='center', va='bottom')
            
            for i, group in enumerate(multi_groups):
                track_ids = group.get('tracking_ids', ['N/A'])
                
                # CHANGE HERE: Use same visual style for all multi-view clusters
                # Whether continued or not
                marker = '*'  # Always use star marker for multi-view clusters
                edge_color = 'white'  # Always use white edge
                
                ax2.scatter(group['centroid'][0], group['centroid'][1],
                        c='black', s=200, alpha=0.8, marker=marker,
                        edgecolors=edge_color, linewidth=2)
                
                track_id_str = ",".join(map(str, track_ids))
                # Don't add the (C) indicator
                ax2.text(group['centroid'][0], group['centroid'][1] + 20,
                    f"IDs:{track_id_str}", fontsize=8, ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7))
                
                for point in group['points']:
                    ax2.plot([group['centroid'][0], point[0]],
                        [group['centroid'][1], point[1]],
                        'gray', alpha=0.6, linewidth=1)
            
            # Update title without mentioning the difference in markers
            ax2.set_title(f'Frame {frame_idx}: Groups (★ = multi-view, ● = single-view)')


 
   
 
    def assign_consistent_tracking_ids(self):
        """
        Assign consistent tracking IDs to points across frames using 
        Hungarian algorithm and motion prediction.
        """
        from scipy.optimize import linear_sum_assignment
        
        print("Assigning consistent tracking IDs across frames...")
        self.tracking_id_last_seen = {}
        next_id = 0
        
        # To store velocity estimates for prediction
        point_velocities = {}
        
        for view_id in sorted(self.view_data.keys()):
            print(f"Processing view {view_id}...")
            prev_points = None
            prev_ids = []
            
            for frame_idx, points in enumerate(self.filtered_positions[view_id]):
                if points is None or len(points) == 0:
                    prev_points = None
                    prev_ids = []
                    continue
                    
                # Extract positions only (first 2 columns)
                current_positions = np.array([(p[0], p[1]) for p in points])
                new_points = []
                
                # First frame in this view - assign new IDs
                if prev_points is None or len(prev_ids) == 0:
                    for i, point in enumerate(points):
                        tracking_id = next_id
                        next_id += 1
                        new_point = (point[0], point[1], tracking_id)
                        new_points.append(new_point)
                        self.tracking_id_last_seen[tracking_id] = frame_idx
                        point_velocities[tracking_id] = np.array([0.0, 0.0])  # Initialize velocity to zero
                else:
                    # For subsequent frames, match with previous frame points
                    # Apply prediction using velocity
                    predicted_positions = []
                    for i, prev_id in enumerate(prev_ids):
                        if prev_id in point_velocities:
                            pred_pos = prev_points[i][:2] + point_velocities[prev_id]
                            predicted_positions.append(pred_pos)
                        else:
                            predicted_positions.append(prev_points[i][:2])
                    
                    predicted_positions = np.array(predicted_positions)
                    
                    # Create cost matrix based on distances between predicted and current positions
                    cost_matrix = np.zeros((len(prev_points), len(current_positions)))
                    #max_distance = self.config.get('max_tracking_distance', 50)  # Max distance for ID maintenance
                    # Use velocity to adjust tracking distance
                    max_distance = self.config.get('max_tracking_distance', 50)
                    if tracking_id in point_velocities:
                        velocity_magnitude = np.linalg.norm(point_velocities[tracking_id])
                        max_distance = max(max_distance, velocity_magnitude * 1.5)


                    
                    for i in range(len(prev_points)):
                        for j in range(len(current_positions)):
                            dist = np.linalg.norm(predicted_positions[i] - current_positions[j])
                            # Penalize matches beyond max_distance
                            cost_matrix[i, j] = dist if dist <= max_distance else 1000000
                    
                    # Use Hungarian algorithm for optimal assignment
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    # Create a map of assigned prev_idx -> current_idx
                    assignments = {}
                    for i, j in zip(row_ind, col_ind):
                        if cost_matrix[i, j] <= max_distance:  # Only use valid assignments
                            assignments[i] = j
                    
                    # Update points with assigned IDs or create new ones
                    assigned_current = set()
                    for prev_idx, curr_idx in assignments.items():
                        tracking_id = prev_ids[prev_idx]
                        assigned_current.add(curr_idx)
                        
                        # Calculate new velocity
                        prev_pos = np.array(prev_points[prev_idx][:2])
                        curr_pos = np.array(current_positions[curr_idx])
                        new_velocity = curr_pos - prev_pos
                        
                        # Update velocity with exponential smoothing
                        alpha = 0.7  # Smoothing factor
                        if tracking_id in point_velocities:
                            point_velocities[tracking_id] = alpha * new_velocity + (1-alpha) * point_velocities[tracking_id]
                        else:
                            point_velocities[tracking_id] = new_velocity
                        
                        # Create new point with consistent ID
                        new_point = (points[curr_idx][0], points[curr_idx][1], tracking_id)
                        new_points.append(new_point)
                        self.tracking_id_last_seen[tracking_id] = frame_idx
                    
                    # Assign new IDs to unmatched current points
                    for i in range(len(points)):
                        if i not in assigned_current:
                            tracking_id = next_id
                            next_id += 1
                            new_point = (points[i][0], points[i][1], tracking_id)
                            new_points.append(new_point)
                            self.tracking_id_last_seen[tracking_id] = frame_idx
                            point_velocities[tracking_id] = np.array([0.0, 0.0])  # Initialize velocity
                
                # Update filtered positions with new tracking IDs
                self.filtered_positions[view_id][frame_idx] = new_points
                prev_points = new_points
                prev_ids = [p[2] for p in new_points]
                
                # Clean up inactive velocities periodically
                if frame_idx % 100 == 0:
                    active_ids = set(prev_ids)
                    ids_to_remove = [vid for vid in point_velocities if vid not in active_ids]
                    for vid in ids_to_remove:
                        if vid in point_velocities:
                            del point_velocities[vid]
        
        print(f"Assigned {next_id} unique tracking IDs across all views")


    
    
    def _update_visualization_functions(self):
       """Update the plot functions to use consistent IDs"""
       # This function would modify plotting/animation to use consistent IDs
       # For now we'll just note that this would be implemented
       print("Visualization updated to use consistent tracking IDs")
 
 
 
    def plot_transformed_images(self):
        fig, axes = plt.subplots(1, len(self.view_data), figsize=(5 * len(self.view_data), 5))
        if len(self.view_data) == 1:
            axes = [axes]
        for idx, view_id in enumerate(sorted(self.view_data.keys())):
            ax = axes[idx]
            img_path = self.view_data[view_id]['image_file']
            if img_path is None:
                ax.set_title(f"View {view_id}: No image")
                ax.axis('off')
                continue
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            warped = cv2.warpPerspective(img, self.transformation_matrices[view_id],
                                         (self.plot_bounds[view_id]['x_max'], self.plot_bounds[view_id]['y_max']))
            ax.imshow(warped)
            ax.set_title(f'Transformed View {view_id}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()
 
    def animate_grouped_centroids(self, frame_skip=1, video_paths=None):
        fig = plt.figure(figsize=(16, 12))
 
        ax_plot = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig)
        bounds = next(iter(self.plot_bounds.values()))
 
        first_view = next(iter(self.view_data))
        corners = self.view_data[first_view]['corners']
        H = self.transformation_matrices[first_view]
        tbl_poly = plt.Polygon(
            self.transform_points(corners, H),
            fill=False, edgecolor='black', linewidth=3
        )
        ax_plot.add_patch(tbl_poly)
        ax_plot.scatter(
            bounds['center_x'], bounds['center_y'],
            c='red', s=100, marker='x', label='Table Center'
        )
 
        group_color = 'purple'
        group_scatter = ax_plot.scatter([], [], s=300, alpha=0.8,
                                      c=group_color, label='Multi-View Groups')
        connection_lines = []
 
        tracking_annotations = []
 
        colors = ['blue', 'green', 'red', 'orange', 'cyan']
        individual_scatters = {}
        for i, view_id in enumerate(sorted(self.view_data.keys())):
            individual_scatters[view_id] = ax_plot.scatter([], [], s=50, alpha=0.6,
                                                        c=colors[i % len(colors)],
                                                        label=f'View {view_id} Individual')
 
        show_individual = [True]
        show_ids = [True]
        show_consistent_ids = [True]  # New toggle for consistent IDs
 
        def toggle_individual(event):
            if event.key == ' ':
                show_individual[0] = not show_individual[0]
                print(f"Individual centroids display: {'ON' if show_individual[0] else 'OFF'}")
            elif event.key == 'i':
                show_ids[0] = not show_ids[0]
                print(f"Tracking ID display: {'ON' if show_ids[0] else 'OFF'}")
            elif event.key == 'c':
                show_consistent_ids[0] = not show_consistent_ids[0]
                print(f"Consistent ID display: {'ON' if show_consistent_ids[0] else 'OFF'}")
 
        fig.canvas.mpl_connect('key_press_event', toggle_individual)
 
        ax_plot.set_xlim(bounds['x_min'], bounds['x_max'])
        ax_plot.set_ylim(bounds['y_min'], bounds['y_max'])
        ax_plot.invert_yaxis()
        #ax_plot.set_title('Multi-View Tracking (SPACE: toggle points, i: toggle IDs, c: toggle consistent IDs)', fontsize=14)
        ax_plot.grid(True, alpha=0.3)
        ax_plot.legend(fontsize=10, loc='upper right')
 
        ax_video1 = plt.subplot2grid((2, 2), (1, 0), fig=fig)
        cap1 = None
        img_handle1 = None
        if video_paths and len(video_paths) > 0 and os.path.exists(video_paths[0]):
            cap1 = cv2.VideoCapture(video_paths[0])
            ret1, frame1 = cap1.read()
            if ret1:
                frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                mirrored1 = cv2.flip(frame1_rgb, 1)
                img_handle1 = ax_video1.imshow(mirrored1)
                ax_video1.axis('off')
                ax_video1.set_title('Mirrored Video - View 4')
 
        ax_video2 = plt.subplot2grid((2, 2), (1, 1), fig=fig)
        cap2 = None
        img_handle2 = None
        if video_paths and len(video_paths) > 1 and os.path.exists(video_paths[1]):
            cap2 = cv2.VideoCapture(video_paths[1])
            ret2, frame2 = cap2.read()
            if ret2:
                frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                mirrored2 = cv2.flip(frame2_rgb, 1)
                img_handle2 = ax_video2.imshow(mirrored2)
                ax_video2.axis('off')
                ax_video2.set_title('Mirrored Video - View 1')
 
        total_video_frames = len(self.clustered_positions)
        if cap1:
            total_video_frames = min(total_video_frames, int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        if cap2:
            total_video_frames = min(total_video_frames, int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

        
        
        
        
        def animate(i):
            f = i * frame_skip
            if f >= total_video_frames:
                return
        
            for line in connection_lines:
                line.remove()
            connection_lines.clear()
        
            for ann in tracking_annotations:
                ann.remove()
            tracking_annotations.clear()
        

            # In animate_grouped_centroids, add this after you get current_groups:
            current_groups = self.clustered_positions.get(f, [])

            # SAFETY CHECK - never display more than max clusters
            if len(current_groups) > self.config['max_clusters_per_frame']:
                # This should never happen, but just to be safe
                current_groups = sorted(current_groups, 
                                    key=lambda g: (-len(g['views']), -len(g['points'])))
                current_groups = current_groups[:self.config['max_clusters_per_frame']]


            
            
            
            if current_groups:
                group_centroids = np.array([g['centroid'] for g in current_groups])
                group_scatter.set_offsets(group_centroids)

                for group in current_groups:
                    for point in group['points']:
                        line = ax_plot.plot([group['centroid'][0], point[0]],
                                    [group['centroid'][1], point[1]],
                                    'gray', alpha=0.6, linewidth=1)[0]
                        connection_lines.append(line)
            else:
                group_scatter.set_offsets(np.empty((0, 2)))
        
            if show_individual[0]:
                for view_id in sorted(self.view_data.keys()):
                    if f < len(self.filtered_positions[view_id]):
                        centroids = self.filtered_positions[view_id][f]
                        if centroids is not None and len(centroids) > 0:
                            points_xy = [(point[0], point[1]) for point in centroids]
                            transformed = self.transform_points(points_xy, self.transformation_matrices[view_id])
                        
                            individual_scatters[view_id].set_offsets(transformed)
                        
                            if show_ids[0]:
                                for j, point in enumerate(transformed):
                                    if j < len(centroids):
                                        original_id = centroids[j][2]
                                        
                                        # Check for consistent ID if available and enabled
                                        if show_consistent_ids[0] and hasattr(self, 'consistent_ids') and f in self.consistent_ids:
                                            key = (view_id, original_id)
                                            if key in self.consistent_ids[f]:
                                                track_id = f"#{self.consistent_ids[f][key]}"
                                            else:
                                                track_id = str(original_id)
                                        else:
                                            track_id = str(original_id)
                                        
                                        ann = ax_plot.text(point[0], point[1] - 15,
                                                    track_id, fontsize=8,
                                                    ha='center', va='bottom',
                                                    color=colors[(view_id - 1) % len(colors)])
                                        tracking_annotations.append(ann)
                        else:
                            individual_scatters[view_id].set_offsets(np.empty((0, 2)))
                    else:
                        individual_scatters[view_id].set_offsets(np.empty((0, 2)))
            else:
                for view_id in individual_scatters:
                    individual_scatters[view_id].set_offsets(np.empty((0, 2)))
        
            id_type = "Consistent IDs" if show_consistent_ids[0] else "Original IDs" 
            individual_status = "Individual: ON" if show_individual[0] else "Individual: OFF"
            ids_status = f"IDs: ON ({id_type})" if show_ids[0] else "IDs: OFF"
            fig.suptitle(f'Frame {f+1}/{total_video_frames} - Groups: {len(current_groups)} - {individual_status} - {ids_status}',
                        fontsize=14)
        
            if cap1 and img_handle1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret_vid1, frame_vid1 = cap1.read()
                if ret_vid1:
                    rgb_vid1 = cv2.cvtColor(frame_vid1, cv2.COLOR_BGR2RGB)
                    mirrored_vid1 = cv2.flip(rgb_vid1, 1)
                    img_handle1.set_data(mirrored_vid1)
        
            if cap2 and img_handle2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret_vid2, frame_vid2 = cap2.read()
                if ret_vid2:
                    rgb_vid2 = cv2.cvtColor(frame_vid2, cv2.COLOR_BGR2RGB)
                    mirrored_vid2 = cv2.flip(rgb_vid2, 1)
                    img_handle2.set_data(mirrored_vid2)


 
        frames = total_video_frames // frame_skip
        anim = FuncAnimation(
            fig,
            animate,
            frames=frames,
            interval=33,
            blit=False,
            repeat=True
        )
 
        plt.tight_layout(pad=2.0)
        plt.show()
 
        if cap1:
            cap1.release()
        if cap2:
            cap2.release()
 
        return anim
    
 
    def reconstruct_continued_clusters(self):
        """Reconstruct continued clusters that might be missing after loading from file"""
        print("Reconstructing continued clusters...")
        
        # Start from second frame
        frame_indices = sorted(self.clustered_positions.keys())
        
        for i in range(1, len(frame_indices)):
            prev_frame_idx = frame_indices[i-1]
            curr_frame_idx = frame_indices[i]
            
            prev_clusters = self.clustered_positions[prev_frame_idx]
            curr_clusters = self.clustered_positions[curr_frame_idx]
            
            # Get all tracking IDs in current clusters
            current_tracking_ids = set()
            for group in curr_clusters:
                for track_id in group.get('tracking_ids', []):
                    current_tracking_ids.add(track_id)
            
            # For each previous cluster, check if it disappeared
            for prev_idx, prev_cluster in enumerate(prev_clusters):
                for track_id in prev_cluster.get('tracking_ids', []):
                    if track_id not in current_tracking_ids:
                        # Look for points with this ID in filtered_positions
                        for view_id in self.filtered_positions:
                            if curr_frame_idx < len(self.filtered_positions[view_id]):
                                centroids = self.filtered_positions[view_id][curr_frame_idx]
                                if centroids is not None:
                                    for point in centroids:
                                        if point[2] == track_id:
                                            # Found a point matching this ID - create continued cluster
                                            new_point = self.transform_points([(point[0], point[1])], 
                                                                            self.transformation_matrices[view_id])[0]
                                            
                                            # Create continued cluster
                                            continued_cluster = {
                                                'centroid': new_point,
                                                'views': [int(view_id)],
                                                'points': [new_point],
                                                'tracking_ids': [track_id],
                                                'continued': True,
                                                'original_centroid': prev_cluster['centroid']
                                            }
                                            
                                            # Add to current frame clusters
                                            curr_clusters.append(continued_cluster)
                                            #print(f"  Reconstructed continued cluster at frame {curr_frame_idx}")
                                            break
        
        print("Finished reconstructing continued clusters")

 
 
    def run_full_pipeline(self):
        print("=== Enhanced Multi-View Person Tracking Pipeline ===")
        self.load_view_data()
        if not self.view_data:
            return
 
        self.filter_static_points()
        self.create_homography_matrices()
 
        num_tracks = self.assign_consistent_tracking_ids()
        print(f"created {num_tracks} consistent tracks letsgooooo")
        self.group_centroids_across_views()
 
        all_frames = []
        for f in range(len(self.clustered_positions)):
            cent_list = [g['centroid'] for g in self.clustered_positions[f]]
            all_frames.append(np.array(cent_list))
        np.save('grouped_centroids.npy', np.array(all_frames, dtype=object))
        print("Saved grouped centroids to 'grouped_centroids.npy'")
 
        print("\n=== Debug plot for frame 0 ===")
        self.plot_debug_frame(frame_idx=0)
        print("\n=== Transformed View Images ===")
        self.plot_transformed_images()
 
        print("\n=== Grouped centroids animation with dual video display ===")
        video_paths = [
            '/home/piyansh/multi_cam_ds/calibration/footage/cam_4.mp4',
            '/home/piyansh/multi_cam_ds/calibration/footage/cam_1.mp4'
        ]
 
        grouped_anim = self.animate_grouped_centroids(frame_skip=1, video_paths=video_paths)
        return grouped_anim
 
 
def save_grouped_centroids(tracker, filename='grouped_centroids_full.npy'):
   np.save(filename, np.array([
       tracker.clustered_positions[f] for f in sorted(tracker.clustered_positions)
   ], dtype=object))
   print(f"Saved full grouped centroids to '{filename}'")


def load_grouped_centroids(tracker, filename='grouped_centroids_full.npy'):
    grouped_data = np.load(filename, allow_pickle=True)
    tracker.clustered_positions = {i: frame_groups for i, frame_groups in enumerate(grouped_data)}
    
    # Ensure tracking_id_last_seen is populated for continued clusters to work
    tracker.tracking_id_last_seen = {}
    
    # Populate tracking_id_last_seen from loaded clusters
    for frame_idx, groups in tracker.clustered_positions.items():
        for group in groups:
            if 'tracking_ids' in group:
                for track_id in group['tracking_ids']:
                    tracker.tracking_id_last_seen[track_id] = frame_idx
    
    print(f"Loaded grouped centroids from '{filename}' and reconstructed tracking state")


if __name__ == "__main__":
   base_path = '/home/piyansh/multi_cam_ds/calibration/calibration'

   config = {
       # Point merging
       'point_merge_threshold': 100000,    # Distance in pixels to merge close points
       'frame_merge_threshold': 0.1,       # Distance threshold for merging CSV files

       # Static filtering
       'static_min_occurrences': 1000,    # Min occurrences for static point
       'static_proximity_threshold': 2,   # Proximity for rounding static points

       # Clustering
       'cluster_distance_threshold': 850, # Max distance for cluster formation
       'min_samples_for_cluster': 2,      # Min points for valid cluster
       'max_views_per_cluster': 4,        # Max views per cluster
       'max_clusters_per_frame': 3,       # Max clusters to track per frame

       # View 2 specific
       'view2_reduced_weight': 0.5,       # Weight for View 2 in 3+ view groups
       'view2_full_weight': 1.0,          # Weight for View 2 in 2-view groups
       'view2_x_bound': -2179,            # View 2 filtering boundary

       # Tracking
       'tracking_memory_frames': 100,     # Frames to remember tracking associations
       'tracking_cleanup_interval': 300,   # How often to clean up inactive IDs

       # Kalman filter tracking
       'max_lost_frames': 30,         # Maximum frames to keep tracking a lost cluster
       'cluster_dt': 1.0,             # Time step for Kalman filter (frames)
       'cluster_process_noise': 0.01, # Process noise (movement uncertainty)
       'cluster_measurement_noise': 20.0, # Measurement noise
       'cluster_match_threshold': 200.0,  # Max distance for matching clusters
       'smooth_trajectories': True,    # Whether to smooth cluster positions

       # Visualization
       'debug_mode': False,               
       'debug_max_frames': 10,           
   }

   tracker = MultivisionTracker(base_path, num_views=4, config=config)

   run_processing = True  #re-process and save grouped centroids

   if run_processing:
       print("\n=== Running Full Processing Pipeline ===")
       tracker.load_view_data()
       tracker.filter_static_points()
       tracker.create_homography_matrices()
       tracker.assign_consistent_tracking_ids()
       tracker.group_centroids_across_views()
       tracker.track_clusters()

       # Save full grouped centroids
       save_grouped_centroids(tracker)

   print("\n=== Loading and Animating ===")
   tracker.load_view_data()
   tracker.filter_static_points()
   tracker.create_homography_matrices()
   tracker.assign_consistent_tracking_ids()
   load_grouped_centroids(tracker)
   tracker.reconstruct_continued_clusters()

   video_paths = [
       '/home/piyansh/multi_cam_ds/calibration/footage/cam_4.mp4',
       '/home/piyansh/multi_cam_ds/calibration/footage/cam_1.mp4'
   ]

   tracker.animate_grouped_centroids(frame_skip=5, video_paths=video_paths)





