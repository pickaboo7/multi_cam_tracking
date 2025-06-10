import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pandas as pd
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
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
 
        # New attributes for tracking
        self.next_merged_id = 200  # Start merged IDs from 200
        self.merged_id_pairs = {}  # Maps original ID pairs to merged IDs
        self.cluster_tracking_ids = {}  # Maps cluster index to set of tracking IDs
        self.tracking_id_last_seen = {}  # Maps tracking ID to frame it was last seen
        self.current_frame = 0  # Current frame being processed
        self.cluster_history = []  # List to store cluster history per frame
 
        # Configuration with defaults
        self.config = {
            # Point merging
            'point_merge_threshold': 10,  # Distance in pixels to merge close points within same view
            'frame_merge_threshold': 15,  # Distance threshold when merging between primary/alt CSV files
 
            # Static filtering
            'static_min_occurrences': 200,  # Minimum occurrences for a point to be considered static
            'static_proximity_threshold': 2,  # Proximity for rounding points when counting
 
            # Clustering
            'cluster_distance_threshold': 700,  # Maximum distance for points to be in the same cluster
            'min_samples_for_cluster': 2,  # Minimum samples required for valid cluster
            'max_views_per_cluster': 4,  # Maximum views included in one cluster
            'max_clusters_per_frame': 3,  # Maximum clusters to track per frame
 
            # View 2 specific
            'view2_reduced_weight': 0.5,  # Weight for View 2 in 3+ view groups
            'view2_full_weight': 1.0,    # Weight for View 2 in 2-view groups
            'view2_x_bound': -2179,      # Left boundary for View 2 point filtering
 
            # Tracking
            'tracking_memory_frames': 150, # How many frames to remember past tracking associations
            'tracking_cleanup_interval': 50, # How often to clean up inactive tracking IDs
 
            # Visualization
            'debug_mode': True,  # Enable debug output
            'debug_max_frames': 10,  # Max frames to show detailed debug info
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
                        # Merge close points within the same view (same point detected in same source but different threshold)
                        merged_points = self._merge_close_points(points, proximity_threshold=self.config['point_merge_threshold'])
                        centroids_list[frame] = merged_points
 
                    csv_centroids[view_id] = centroids_list
                    print(f"    Loaded primary CSV with {len(df)} entries across {len(frames_grouped)} frames")
                except Exception as e:
                    print(f"     Error loading CSV: {e}")
            else:
                print(f"     CSV file missing!")
 
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
                    print(f"    Loaded alternative CSV with {len(df)} entries across {len(frames_grouped)} frames")
                except Exception as e:
                    print(f"    Error loading alternative CSV: {e}")
            else:
                print(f"     Alternative CSV file missing!")
 
        for view_id in range(1, self.num_views + 1):
            corners_file = f'{self.base_path}/points_cam_{view_id}.npy'
            image_file = f'{os.path.dirname(self.base_path)}/frame_cam_{view_id}.jpg'
 
            print(f"Looking for View {view_id} calibration data:")
            print(f"  Corners: {corners_file}")
            print(f"  Image: {image_file}")
 
            if not os.path.exists(corners_file):
                print(f"   Corners file missing!")
                continue
 
            if view_id-1 not in csv_centroids and view_id-1 not in alt_csv_centroids:
                print(f"  No centroids data available!")
                continue
 
            if not os.path.exists(image_file):
                print(f"   Image file missing (optional)")
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
 
                print(f"  View {view_id}: {len(combined_centroids)} total frames, {non_empty_frames} non-empty frames")
                print(f"     Combined {total_points} total points")
                print(f"     Corners shape: {corners.shape}")
            except Exception as e:
                print(f"  Error loading view {view_id}: {e}")
 
        print(f"\nSuccessfully loaded {len(self.view_data)} views")
 
    def _merge_close_points(self, points, proximity_threshold=10):
        """Merge points that are within proximity_threshold distance (logic not required anymore)."""
        return points
 
    def filter_static_points(self, min_occurrences=None, proximity_threshold=None):
        """Filter out static points based on minimum occurrences and proximity threshold, these are false detection (stationary objects or noise)."""
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
        # If points are tuples/lists with more than 2 elements, extract just the x,y parts
        if isinstance(points[0], (tuple, list)) and len(points[0]) > 2:
            pts = np.array([(p[0], p[1]) for p in points], dtype=np.float32).reshape(-1, 1, 2)
        else:
            pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
 
        warped = cv2.perspectiveTransform(pts, H)
        return warped.reshape(-1, 2)
 
    def debug_transformed_points(self, frame_idx=0, max_debug_frames=None):
        """Debugging function to print transformed points for a specific frame."""
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
        Modified grouping logic:
        1. First use spatial proximity to form clusters (primary approach)
        2. Then check if any clusters from previous frame are missing
        3. If missing clusters found, try to continue them using single points
        """
        if eps is None:
            eps = self.config['cluster_distance_threshold']
        if min_samples is None:
            min_samples = self.config['min_samples_for_cluster']
        if debug_mode is None:
            debug_mode = self.config['debug_mode']
    
        print(f"Grouping centroids across views with eps={eps}...")
        print(f"View 2 weightage: {self.config['view2_reduced_weight']} weight in 3-4 view groups, {self.config['view2_full_weight']} weight in 2-view groups")
        print(f"View 2 filtering: Points with x < {self.config['view2_x_bound']} after transformation will be removed")
    
        min_frames = min(len(self.filtered_positions[view_id]) for view_id in self.filtered_positions)
    
        # Initialize cluster history
        self.cluster_history = [[] for _ in range(min_frames)]
    
        for frame_idx in range(min_frames):
            self.current_frame = frame_idx
    
            # Debug first few frames
            if debug_mode and frame_idx < self.config['debug_max_frames']:
                self.debug_transformed_points(frame_idx)
    
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
                        valid_x_mask = (transformed_pts[:, 0] <= 4000) & (transformed_pts[:, 0] >= -1700)
                        transformed_pts = transformed_pts[valid_x_mask]
                        filtered_tracking_ids = [frame_tracking_ids[i] for i, valid in enumerate(valid_x_mask) if valid]
    
                        if len(transformed_pts) > 0:  # Only add if points remain after filtering
                            all_points.extend(transformed_pts)
                            view_ids.extend([view_id] * len(transformed_pts))
                            tracking_ids.extend(filtered_tracking_ids)
    
            if len(all_points) == 0:
                self.clustered_positions[frame_idx] = []
                continue
    
            all_points = np.array(all_points)
            view_ids = np.array(view_ids)
    
            if len(all_points) < 2:
                self.clustered_positions[frame_idx] = [{
                    'centroid': all_points[0],
                    'views': [int(view_ids[0])],
                    'points': [all_points[0]],
                    'tracking_ids': [tracking_ids[0]]
                }]
    
                # Update tracking ID last seen
                self.tracking_id_last_seen[tracking_ids[0]] = frame_idx
                continue
    
            # Use greedy approach to form groups with closest points
            final_groups = []
            used_points = set()
    
            # FIRST PRIORITY: Spatial clustering
            # Start with each point as a potential group center for remaining points
            for i, center_point in enumerate(all_points):
                if i in used_points:
                    continue
    
                center_view = view_ids[i]
                center_track_id = tracking_ids[i]
    
                # Find all points within eps distance from different views
                distances = np.linalg.norm(all_points - center_point, axis=1)
                nearby_mask = (distances <= eps) & (view_ids != center_view)
    
                if not np.any(nearby_mask):
                    # No nearby points from other views, create single-view group
                    if i not in used_points:
                        final_groups.append({
                            'centroid': center_point,
                            'views': [int(center_view)],
                            'points': [center_point],
                            'tracking_ids': [center_track_id]
                        })
                        used_points.add(i)
    
                        # Update tracking ID last seen
                        self.tracking_id_last_seen[center_track_id] = frame_idx
                    continue
    
                # Get nearby points from other views
                nearby_points = all_points[nearby_mask]
                nearby_views = view_ids[nearby_mask]
                nearby_indices = np.where(nearby_mask)[0]
                nearby_track_ids = [tracking_ids[j] for j in nearby_indices]
    
                # Select closest point from each nearby view
                group_points = [center_point]
                group_views = [center_view]
                group_indices = [i]
                group_track_ids = [center_track_id]
    
                unique_nearby_views = np.unique(nearby_views)
    
                # Add points from all views including view 2
                for view_id in unique_nearby_views:
                    if len(group_views) >= self.config['max_views_per_cluster']:  # Max views per group
                        break
    
                    view_mask = nearby_views == view_id
                    view_points = nearby_points[view_mask]
                    view_indices = nearby_indices[view_mask]
                    view_track_ids = [nearby_track_ids[j] for j, is_view in enumerate(view_mask) if is_view]
    
                    # Find closest point from this view
                    view_distances = np.linalg.norm(view_points - center_point, axis=1)
                    closest_idx = np.argmin(view_distances)
                    closest_point_idx = view_indices[closest_idx]
                    closest_track_id = view_track_ids[closest_idx]
    
                    if closest_point_idx not in used_points:
                        group_points.append(view_points[closest_idx])
                        group_views.append(view_id)
                        group_indices.append(closest_point_idx)
                        group_track_ids.append(closest_track_id)
    
                # Only create multi-view group if we have points from different views
                if len(group_views) > 1:
                    # Calculate weighted centroid - reduce view 2 weight for 3-4 point groups
                    if len(group_views) >= 3:
                        # Apply weights: view 2 gets reduced weight, others get full weight
                        weights = []
                        weighted_points = []
    
                        for i, (point, view_id) in enumerate(zip(group_points, group_views)):
                            if view_id == 2:
                                weight = self.config['view2_reduced_weight']  # Custom weight for view 2
                                if debug_mode and frame_idx < self.config['debug_max_frames']:
                                    print(f"    Reducing View 2 weight to {weight} in {len(group_views)}-view group")
                            else:
                                weight = self.config['view2_full_weight']  # Full weight for other views
    
                            weights.append(weight)
                            weighted_points.append(point * weight)
    
                        # Calculate weighted average
                        weights = np.array(weights)
                        weighted_points = np.array(weighted_points)
                        weighted_centroid = np.sum(weighted_points, axis=0) / np.sum(weights)
                    else:
                        # For 2-view groups, use regular mean
                        weighted_centroid = np.mean(group_points, axis=0)
    
                    final_groups.append({
                        'centroid': weighted_centroid,
                        'views': [int(view) for view in group_views],  # Ensure views are integers
                        'points': group_points,
                        'tracking_ids': group_track_ids,
                        'weights': weights if len(group_views) >= 3 else [1.0] * len(group_views)
                    })
    
                    # Mark all points in this group as used
                    for idx in group_indices:
                        used_points.add(idx)
    
                    # Update tracking ID last seen
                    for track_id in group_track_ids:
                        self.tracking_id_last_seen[track_id] = frame_idx
                elif i not in used_points:
                    # Single view group
                    final_groups.append({
                        'centroid': center_point,
                        'views': [int(center_view)],
                        'points': [center_point],
                        'tracking_ids': [center_track_id]
                    })
                    used_points.add(i)
    
                    # Update tracking ID last seen
                    self.tracking_id_last_seen[center_track_id] = frame_idx
    
            # SECOND PRIORITY: Check for missing clusters from previous frame
            # SECOND PRIORITY: Check for missing clusters from previous frame
            if frame_idx > 0 and frame_idx - 1 in self.clustered_positions:
                prev_clusters = self.clustered_positions[frame_idx - 1]
                current_tracking_ids = set()
                
                # Get all tracking IDs in current clusters
                for group in final_groups:
                    for track_id in group['tracking_ids']:
                        current_tracking_ids.add(track_id)
                
                # Track which clusters we've already continued
                continued_clusters = set()
                
                # Check each previous cluster
                for prev_cluster_idx, prev_cluster in enumerate(prev_clusters):
                    # Skip if this cluster is already continued
                    if prev_cluster_idx in continued_clusters:
                        continue
                        
                    # Check if any tracking IDs from this cluster are missing in current clusters
                    missing_ids = []
                    for track_id in prev_cluster.get('tracking_ids', []):
                        if track_id not in current_tracking_ids:
                            missing_ids.append(track_id)
                    
                    # If we have missing IDs, look for matching points - INCLUDING USED POINTS
                    if missing_ids:
                        # Look for ANY points with matching IDs, even if already used in other clusters
                        matching_points_indices = []
                        
                        for i, track_id in enumerate(tracking_ids):
                            if track_id in missing_ids:
                                matching_points_indices.append(i)
                        
                        # If we found any matching points (even used ones), create a continued cluster
                        if matching_points_indices:
                            # Use first matched point as cluster center
                            idx = matching_points_indices[0]
                            track_id = tracking_ids[idx]
                            view_id = view_ids[idx]
                            point = all_points[idx]
                            
                            # Create new continued cluster
                            continued_cluster = {
                                'centroid': point,
                                'views': [int(view_id)],
                                'points': [point],
                                'tracking_ids': [track_id],
                                'continued': True,  # Mark as continued
                                'continued_from_cluster': prev_cluster_idx,  # Remember which cluster this continues
                                'original_centroid': prev_cluster['centroid']  # Store original position for smoother transitions
                            }
                            
                            # Add other matching points from different views if available
                            for other_idx in matching_points_indices[1:]:
                                # Only add points from different views to avoid duplicates
                                if view_ids[other_idx] != view_id and view_ids[other_idx] not in continued_cluster['views']:
                                    continued_cluster['views'].append(int(view_ids[other_idx]))
                                    continued_cluster['points'].append(all_points[other_idx])
                                    continued_cluster['tracking_ids'].append(tracking_ids[other_idx])
                            
                            # Add to final groups
                            final_groups.append(continued_cluster)
                            continued_clusters.add(prev_cluster_idx)
                            
                            # Update tracking ID last seen for all IDs in this continued cluster
                            for tid in continued_cluster['tracking_ids']:
                                self.tracking_id_last_seen[tid] = frame_idx
                                
                            if debug_mode and frame_idx < self.config['debug_max_frames']:
                                print(f"  CONTINUED cluster with tracking_ids: {continued_cluster['tracking_ids']} (from missing cluster {prev_cluster_idx})")
                                print(f"    Used points from views: {continued_cluster['views']}")
    

    
            # Add any remaining unused points as single-view groups
            for i, point in enumerate(all_points):
                if i not in used_points:
                    track_id = tracking_ids[i]
                    final_groups.append({
                        'centroid': point,
                        'views': [int(view_ids[i])],
                        'points': [point],
                        'tracking_ids': [track_id]
                    })
    
                    # Update tracking ID last seen
                    self.tracking_id_last_seen[track_id] = frame_idx
    
            # Limit to at most max_clusters_per_frame groups (prefer groups with most views)
            # Note: Now 'continued' clusters don't get special priority - multi-view groups do
            sorted_groups = sorted(final_groups, key=lambda g: (-len(g['views']), -len(g['points']), g.get('continued', False)))
            self.clustered_positions[frame_idx] = sorted_groups[:self.config['max_clusters_per_frame']]
    
            # Clean up tracking IDs that haven't been seen for tracking_memory_frames
            if frame_idx % self.config['tracking_cleanup_interval'] == 0:
                self._cleanup_inactive_tracking_ids(frame_idx, inactive_threshold=self.config['tracking_memory_frames'])
    
            # Enhanced debugging output
            multi_groups = [g for g in final_groups if len(g['views']) > 1]
            view2_groups = [g for g in multi_groups if 2 in g['views']]
            view2_weighted_groups = [g for g in multi_groups if len(g['views']) >= 3 and 2 in g['views']]
            continued_groups = [g for g in final_groups if g.get('continued', False)]
    
            # Record cluster history for this frame
            self.cluster_history[frame_idx] = [group.get('tracking_ids', []) for group in self.clustered_positions[frame_idx]]
    
            if debug_mode and frame_idx < self.config['debug_max_frames']:
                print(f"Frame {frame_idx}: {len(final_groups)} total groups, {len(multi_groups)} multi-view groups, {len(continued_groups)} continued")
    
                # Print tracking IDs in each cluster
                for i, group in enumerate(self.clustered_positions[frame_idx]):
                    track_ids_str = ", ".join(map(str, group.get('tracking_ids', [])))
                    is_continued = group.get('continued', False)
                    print(f"  Cluster {i}: views {group['views']} tracking_ids: [{track_ids_str}] {'(CONTINUED)' if is_continued else ''}")
    
                if view2_groups:
                    print(f"  View 2 included in {len(view2_groups)} groups:")
                    for i, group in enumerate(view2_groups):
                        weight_status = "reduced weight" if len(group['views']) >= 3 else "full weight"
                        print(f"    Group {i}: views {group['views']} ({len(group['views'])} total, {weight_status})")
    
                if view2_weighted_groups:
                    print(f"  View 2 with reduced weight in {len(view2_weighted_groups)} groups:")
                    for i, group in enumerate(view2_weighted_groups):
                        weights_info = dict(zip(group['views'], group.get('weights', [])))
                        print(f"    Group {i}: views {group['views']} weights: {weights_info}")
    
                if multi_groups:
                    for i, group in enumerate(multi_groups):
                        view2_weight = f"View2@{self.config['view2_reduced_weight']}x" if (len(group['views']) >= 3 and 2 in group['views']) else f"View2@{self.config['view2_full_weight']}x" if 2 in group['views'] else "no View2"
                        print(f"  Multi-group {i}: views {group['views']} ({view2_weight})")
    
            if frame_idx % 100 == 0 and frame_idx > 0:
                view2_multi_groups = len([g for g in multi_groups if 2 in g['views']])
                view2_reduced_groups = len([g for g in multi_groups if len(g['views']) >= 3 and 2 in g['views']])
                print(f"Frame {frame_idx}: {len(multi_groups)} multi-view groups ({view2_multi_groups} with View 2, {view2_reduced_groups} with reduced View 2 weight)")
    
                # Print tracking ID statistics
                active_tracking_ids = len(self.tracking_id_last_seen)
                merged_ids = len([tid for tid in self.tracking_id_last_seen if tid >= 200])
                print(f"  Active tracking IDs: {active_tracking_ids} ({merged_ids} merged)")
 
 
 
 
 
    def _cleanup_inactive_tracking_ids(self, current_frame, inactive_threshold=None):
        """ Cleanup tracking IDs that haven't been seen for a certain number of frames to clear up space and allow re-allocation"""
        if inactive_threshold is None:
            inactive_threshold = self.config['tracking_memory_frames']
 
        inactive_ids = []
 
        for track_id, last_seen in self.tracking_id_last_seen.items():
            if current_frame - last_seen > inactive_threshold:
                inactive_ids.append(track_id)
 
        for track_id in inactive_ids:
            del self.tracking_id_last_seen[track_id]
 
        if inactive_ids:
            print(f"Frame {current_frame}: Released {len(inactive_ids)} tracking IDs that weren't seen for {inactive_threshold} frames")
 
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
         Assigns consistent tracking IDs to centroids after homography transformation
         but before clustering, based on spatial proximity between frames.
         """
         print("Assigning consistent tracking IDs to transformed centroids...")
 
         # Dictionary of active trackers {tracker_id: {'position': [x,y], 'last_seen': frame_idx}}
         active_trackers = {}
         next_tracker_id = 1000  # Start with ID 1000 to distinguish from original IDs
 
         # Dictionary to store consistent IDs for visualization
         self.consistent_ids = {}
 
         # Distance threshold for continuing a track
         distance_threshold = self.config['cluster_distance_threshold'] * 0.6
 
         # Process frames sequentially
         min_frames = min(len(self.filtered_positions[view_id]) for view_id in self.filtered_positions)
 
         for frame_idx in range(min_frames):
             self.consistent_ids[frame_idx] = {}
 
             # Get all transformed points for this frame
             all_transformed_points = []
             all_original_ids = []
             all_view_ids = []
 
             for view_id in sorted(self.view_data.keys()):
                 if frame_idx < len(self.filtered_positions[view_id]):
                     centroids = self.filtered_positions[view_id][frame_idx]
                     if centroids is not None and len(centroids) > 0:
                         # Extract only x,y coordinates for transformation
                         points_xy = [(point[0], point[1]) for point in centroids]
                         transformed = self.transform_points(points_xy, self.transformation_matrices[view_id])
 
                         # Filter View 2 points if needed
                         if view_id == 2:
                             valid_mask = transformed[:, 0] >= self.config['view2_x_bound']
                             transformed = transformed[valid_mask]
                             valid_centroids = [centroids[i] for i, valid in enumerate(valid_mask) if valid]
                         else:
                             valid_centroids = centroids
 
                         # Store transformed points with metadata
                         for i, point in enumerate(transformed):
                             all_transformed_points.append(point)
                             all_original_ids.append(valid_centroids[i][2])  # Original tracking ID
                             all_view_ids.append(view_id)
 
             # Skip if no points in this frame
             if not all_transformed_points:
                 continue
 
             all_transformed_points = np.array(all_transformed_points)
 
             # Match with existing trackers
             assigned_trackers = set()
 
             # First, try to continue existing tracks
             for i, point in enumerate(all_transformed_points):
                 original_id = all_original_ids[i]
                 view_id = all_view_ids[i]
 
                 best_tracker_id = None
                 best_distance = float('inf')
 
                 # Find closest tracker
                 for tracker_id, tracker_info in active_trackers.items():
                     if tracker_id in assigned_trackers:
                         continue
 
                     # Calculate distance to last position
                     distance = np.linalg.norm(point - tracker_info['position'])
 
                     # Only assign if within threshold and closest
                     if distance < distance_threshold and distance < best_distance:
                         best_distance = distance
                         best_tracker_id = tracker_id
 
                 # Assign best tracking ID or create new one
                 if best_tracker_id is not None:
                     # Update existing tracker
                     active_trackers[best_tracker_id]['position'] = point
                     active_trackers[best_tracker_id]['last_seen'] = frame_idx
                     assigned_trackers.add(best_tracker_id)
 
                     # Store mapping from original_id to consistent_id for this frame
                     self.consistent_ids[frame_idx][(view_id, original_id)] = best_tracker_id
                 else:
                     # Create new tracker
                     active_trackers[next_tracker_id] = {
                         'position': point,
                         'last_seen': frame_idx
                     }
 
                     # Store mapping
                     self.consistent_ids[frame_idx][(view_id, original_id)] = next_tracker_id
                     next_tracker_id += 1
 
             # Clean up inactive trackers
             inactive_trackers = []
             for tracker_id, info in active_trackers.items():
                 if frame_idx - info['last_seen'] > self.config['tracking_memory_frames']:
                     inactive_trackers.append(tracker_id)
 
             for tracker_id in inactive_trackers:
                 del active_trackers[tracker_id]
 
         print(f"Assigned {next_tracker_id-1000} consistent tracking IDs")
         return next_tracker_id-1000
  
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
        ax_plot.set_title('Multi-View Tracking (SPACE: toggle points, i: toggle IDs, c: toggle consistent IDs)', fontsize=14)
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
        
            current_groups = self.clustered_positions.get(f, [])
            multi_view_groups = [g for g in current_groups if len(g['views']) > 1]
        
            if multi_view_groups:
                group_centroids = np.array([g['centroid'] for g in multi_view_groups])
                group_scatter.set_offsets(group_centroids)
            
                if show_ids[0]:
                    for j, group in enumerate(multi_view_groups):
                        if 'tracking_ids' in group:
                            # Use consistent IDs if available and enabled
                            if show_consistent_ids[0] and hasattr(self, 'consistent_ids') and f in self.consistent_ids:
                                # Try to find consistent IDs for the group's tracking IDs
                                consistent_ids = []
                                for view_id, track_id in zip(group['views'], group['tracking_ids']):
                                    key = (view_id, track_id)
                                    if key in self.consistent_ids[f]:
                                        consistent_ids.append(str(self.consistent_ids[f][key]))
                                
                                if consistent_ids:
                                    track_ids_str = "#" + ",".join(consistent_ids)
                                else:
                                    track_ids_str = ",".join(map(str, group['tracking_ids']))
                            else:
                                track_ids_str = ",".join(map(str, group['tracking_ids']))
                            
                            # No more "(C)" indicator for continued clusters
                            ann = ax_plot.text(group['centroid'][0], group['centroid'][1] + 25,
                                        f"ID:{track_ids_str}", fontsize=9,
                                        ha='center', color='white',
                                        bbox=dict(facecolor='black', alpha=0.7))
                            tracking_annotations.append(ann)
            
                for group in multi_view_groups:
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
            fig.suptitle(f'Frame {f+1}/{total_video_frames} - Groups: {len(multi_view_groups)} - {individual_status} - {ids_status}',
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
   print(f"Loaded grouped centroids from '{filename}'")


if __name__ == "__main__":
   base_path = '/home/piyansh/multi_cam_ds/calibration/calibration'

   # Central configuration for tunable parameters
   config = {
       # Point merging
       'point_merge_threshold': 100000,       # Distance in pixels to merge close points
       'frame_merge_threshold': 5,       # Distance threshold for merging CSV files

       # Static filtering
       'static_min_occurrences': 400,     # Min occurrences for static point
       'static_proximity_threshold': 2,   # Proximity for rounding static points

       # Clustering
       'cluster_distance_threshold': 800, # Max distance for cluster formation
       'min_samples_for_cluster': 2,      # Min points for valid cluster
       'max_views_per_cluster': 4,        # Max views per cluster
       'max_clusters_per_frame': 3,       # Max clusters to track per frame

       # View 2 specific
       'view2_reduced_weight': 0.5,       # Weight for View 2 in 3+ view groups
       'view2_full_weight': 1.0,          # Weight for View 2 in 2-view groups
       'view2_x_bound': -2179,            # View 2 filtering boundary

       # Tracking
       'tracking_memory_frames': 200,     # Frames to remember tracking associations
       'tracking_cleanup_interval': 100,   # How often to clean up inactive IDs

       # Visualization
       'debug_mode': True,               # Show debug output
       'debug_max_frames': 10,           # How many frames to debug in detail
   }

   tracker = MultivisionTracker(base_path, num_views=4, config=config)

   # === STEP 1: PROCESSING PIPELINE ===
   run_processing = True  # Set this to True to re-process and save grouped centroids

   if run_processing:
       print("\n=== Running Full Processing Pipeline ===")
       tracker.load_view_data()
       tracker.filter_static_points()
       tracker.create_homography_matrices()
       tracker.assign_consistent_tracking_ids()
       tracker.group_centroids_across_views()

       # Save full grouped centroids
       save_grouped_centroids(tracker)

   # === STEP 2: LOAD AND ANIMATE ===
   print("\n=== Loading and Animating ===")
   tracker.load_view_data()
   tracker.create_homography_matrices()
   tracker.assign_consistent_tracking_ids()
   load_grouped_centroids(tracker)

   video_paths = [
       '/home/piyansh/multi_cam_ds/calibration/footage/cam_4.mp4',
       '/home/piyansh/multi_cam_ds/calibration/footage/cam_1.mp4'
   ]

   tracker.animate_grouped_centroids(frame_skip=1, video_paths=video_paths)
