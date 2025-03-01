import os
import numpy as np
import pandas as pd
import glob
import cv2
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import osmnx as ox
import geopandas as gpd
from pyproj import Transformer

class DatasetPreparator:
    def __init__(self, data_root, output_dir):
        """
        Initialize the dataset preparation pipeline
        
        Args:
            data_root (str): Path to raw sensor data
            output_dir (str): Where to save the processed dataset
        """
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        self.img_dir = os.path.join(output_dir, 'images')
        self.lidar_dir = os.path.join(output_dir, 'lidar')
        self.meta_dir = os.path.join(output_dir, 'metadata')
        
        for dir_path in [self.img_dir, self.lidar_dir, self.meta_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Camera and LiDAR calibration parameters
        self.calib = None
        self.load_calibration()
        
    def load_calibration(self, calib_file=None):
        """
        Load camera and LiDAR calibration parameters from file or use defaults
        """
        if calib_file and os.path.exists(calib_file):
            # Load calibration from file
            self.calib = np.load(calib_file, allow_pickle=True).item()
        else:
            # Default calibration (sample values, should be replaced with actual calibration)
            self.calib = {
                'camera_matrix': np.array([
                    [1000, 0, 960],
                    [0, 1000, 540],
                    [0, 0, 1]
                ]),
                'distortion_coeffs': np.zeros(5),
                'lidar_to_camera': np.eye(4),  # 4x4 transformation matrix
                'camera_to_lidar': np.eye(4)
            }
            print("Using default calibration. For accurate results, provide actual calibration parameters.")
    
    def find_sequence_data(self, sequence_id):
        """
        Locate all files for a specific sequence
        
        Args:
            sequence_id (str): Identifier for the driving sequence
            
        Returns:
            dict: Paths to all data files for this sequence
        """
        sequence_data = {
            'images': sorted(glob.glob(os.path.join(self.data_root, sequence_id, 'camera', '*.jpg'))),
            'lidar': sorted(glob.glob(os.path.join(self.data_root, sequence_id, 'lidar', '*.pcd'))),
            'gps': os.path.join(self.data_root, sequence_id, 'gps.csv'),
            'timestamps': os.path.join(self.data_root, sequence_id, 'timestamps.csv')
        }
        
        # Verify we have all the necessary files
        for key, value in sequence_data.items():
            if isinstance(value, list) and not value:
                print(f"Warning: No {key} files found for sequence {sequence_id}")
            elif isinstance(value, str) and not os.path.exists(value):
                print(f"Warning: {key} file not found for sequence {sequence_id}")
                
        return sequence_data
    
    def extract_timestamps(self, sequence_data):
        """
        Extract and align timestamps from all sensors
        
        Args:
            sequence_data (dict): Paths to sequence data files
            
        Returns:
            pd.DataFrame: Aligned timestamps for all sensors
        """
        # Load timestamps
        if os.path.exists(sequence_data['timestamps']):
            timestamps_df = pd.read_csv(sequence_data['timestamps'])
        else:
            # If no timestamp file, we'll have to infer from filenames or create synthetic ones
            print("No timestamp file found, creating synthetic timestamps")
            camera_files = sequence_data['images']
            lidar_files = sequence_data['lidar']
            
            # Create synthetic timestamps at 10Hz
            camera_timestamps = np.arange(len(camera_files)) * 0.1
            lidar_timestamps = np.arange(len(lidar_files)) * 0.1
            
            timestamps_df = pd.DataFrame({
                'timestamp': sorted(list(camera_timestamps) + list(lidar_timestamps)),
                'sensor': ['camera'] * len(camera_timestamps) + ['lidar'] * len(lidar_timestamps),
                'filename': camera_files + lidar_files
            })
        
        return timestamps_df
    
    def interpolate_timestamps(self, timestamps_df):
        """
        Handle timestamp misalignments by interpolation
        
        Args:
            timestamps_df (pd.DataFrame): Raw timestamps
            
        Returns:
            pd.DataFrame: Synchronized timestamps
        """
        # Group by sensor type
        camera_df = timestamps_df[timestamps_df['sensor'] == 'camera']
        lidar_df = timestamps_df[timestamps_df['sensor'] == 'lidar']
        
        # Create master timestamp list (union of all sensor timestamps)
        master_timestamps = sorted(np.unique(timestamps_df['timestamp']))
        
        # For each sensor, find the closest timestamp or interpolate
        result_rows = []
        for ts in master_timestamps:
            row = {'master_timestamp': ts}
            
            # Find or interpolate camera timestamp
            if ts in camera_df['timestamp'].values:
                cam_idx = camera_df[camera_df['timestamp'] == ts].index[0]
                row['camera_file'] = camera_df.loc[cam_idx, 'filename']
                row['camera_timestamp'] = ts
            else:
                # Find closest timestamps before and after
                before = camera_df[camera_df['timestamp'] < ts]
                after = camera_df[camera_df['timestamp'] > ts]
                
                if len(before) > 0 and len(after) > 0:
                    before_idx = before.index[-1]
                    after_idx = after.index[0]
                    
                    # Linear interpolation - here we just take the closest one
                    if ts - camera_df.loc[before_idx, 'timestamp'] < camera_df.loc[after_idx, 'timestamp'] - ts:
                        row['camera_file'] = camera_df.loc[before_idx, 'filename']
                        row['camera_timestamp'] = camera_df.loc[before_idx, 'timestamp']
                    else:
                        row['camera_file'] = camera_df.loc[after_idx, 'filename']
                        row['camera_timestamp'] = camera_df.loc[after_idx, 'timestamp']
                elif len(before) > 0:
                    # Use last available
                    before_idx = before.index[-1]
                    row['camera_file'] = camera_df.loc[before_idx, 'filename']
                    row['camera_timestamp'] = camera_df.loc[before_idx, 'timestamp']
                elif len(after) > 0:
                    # Use first available
                    after_idx = after.index[0]
                    row['camera_file'] = camera_df.loc[after_idx, 'filename']
                    row['camera_timestamp'] = camera_df.loc[after_idx, 'timestamp']
                else:
                    row['camera_file'] = None
                    row['camera_timestamp'] = None
            
            # Do the same for LiDAR (code similar to camera interpolation)
            if ts in lidar_df['timestamp'].values:
                lidar_idx = lidar_df[lidar_df['timestamp'] == ts].index[0]
                row['lidar_file'] = lidar_df.loc[lidar_idx, 'filename']
                row['lidar_timestamp'] = ts
            else:
                before = lidar_df[lidar_df['timestamp'] < ts]
                after = lidar_df[lidar_df['timestamp'] > ts]
                
                if len(before) > 0 and len(after) > 0:
                    before_idx = before.index[-1]
                    after_idx = after.index[0]
                    
                    if ts - lidar_df.loc[before_idx, 'timestamp'] < lidar_df.loc[after_idx, 'timestamp'] - ts:
                        row['lidar_file'] = lidar_df.loc[before_idx, 'filename']
                        row['lidar_timestamp'] = lidar_df.loc[before_idx, 'timestamp']
                    else:
                        row['lidar_file'] = lidar_df.loc[after_idx, 'filename']
                        row['lidar_timestamp'] = lidar_df.loc[after_idx, 'timestamp']
                elif len(before) > 0:
                    before_idx = before.index[-1]
                    row['lidar_file'] = lidar_df.loc[before_idx, 'filename']
                    row['lidar_timestamp'] = lidar_df.loc[before_idx, 'timestamp']
                elif len(after) > 0:
                    after_idx = after.index[0]
                    row['lidar_file'] = lidar_df.loc[after_idx, 'filename']
                    row['lidar_timestamp'] = lidar_df.loc[after_idx, 'timestamp']
                else:
                    row['lidar_file'] = None
                    row['lidar_timestamp'] = None
            
            result_rows.append(row)
        
        return pd.DataFrame(result_rows)
    
    def fetch_osm_data(self, gps_file):
        """
        Fetch OpenStreetMap data for the driving area
        
        Args:
            gps_file (str): Path to GPS coordinates file
            
        Returns:
            dict: OpenStreetMap data for the area
        """
        # Load GPS data
        gps_df = pd.read_csv(gps_file)
        
        # Get bounding box for the GPS coordinates
        north = gps_df['latitude'].max() + 0.01
        south = gps_df['latitude'].min() - 0.01
        east = gps_df['longitude'].max() + 0.01
        west = gps_df['longitude'].min() - 0.01
        
        # Download OSM data for this area
        try:
            print(f"Fetching OSM data for area: N={north}, S={south}, E={east}, W={west}")
            osm_graph = ox.graph_from_bbox(north, south, east, west, network_type='drive')
            osm_nodes, osm_edges = ox.graph_to_gdfs(osm_graph)
            
            # Convert to projected coordinates (metric)
            osm_nodes_proj = osm_nodes.to_crs(epsg=3857)
            osm_edges_proj = osm_edges.to_crs(epsg=3857)
            
            # Prepare transformer to convert GPS to projected coordinates
            transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
            
            # Transform GPS coordinates to the same projection
            x, y = transformer.transform(gps_df['longitude'].values, gps_df['latitude'].values)
            gps_df['x'] = x
            gps_df['y'] = y
            
            return {
                'nodes': osm_nodes_proj,
                'edges': osm_edges_proj,
                'gps_projected': gps_df
            }
            
        except Exception as e:
            print(f"Error fetching OSM data: {e}")
            return None
    
    def process_image(self, image_path):
        """
        Process camera image (undistortion, normalization)
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Processed image
        """
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Undistort using calibration parameters
        undistorted = cv2.undistort(
            img, 
            self.calib['camera_matrix'], 
            self.calib['distortion_coeffs']
        )
        
        # Convert to RGB (OpenCV loads as BGR)
        rgb_img = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
        
        return rgb_img
    
    def process_lidar(self, lidar_path):
        """
        Process LiDAR point cloud
        
        Args:
            lidar_path (str): Path to the LiDAR point cloud file
            
        Returns:
            np.ndarray: Processed point cloud
        """
        # Read point cloud
        try:
            pcd = o3d.io.read_point_cloud(lidar_path)
            points = np.asarray(pcd.points)
            
            if points.shape[0] == 0:
                print(f"Error: Empty point cloud in {lidar_path}")
                return None
            
            # Remove points that are too far away (optional)
            max_distance = 100.0  # meters
            distances = np.sqrt(np.sum(points**2, axis=1))
            points = points[distances < max_distance]
            
            # If point cloud has intensity values, keep them
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                intensities = colors[distances < max_distance, 0]  # Assuming intensity is stored in red channel
                return np.column_stack((points, intensities))
            
            return points
            
        except Exception as e:
            print(f"Error processing LiDAR data: {e}")
            return None
    
    def align_map_data(self, osm_data, sync_df):
        """
        Align GPS and map data with sensor data
        
        Args:
            osm_data (dict): OpenStreetMap data
            sync_df (pd.DataFrame): Synchronized sensor timestamps
            
        Returns:
            pd.DataFrame: DataFrame with aligned map data
        """
        if osm_data is None:
            print("No OSM data available for alignment")
            return sync_df
        
        gps_df = osm_data['gps_projected']
        
        # For each master timestamp, find the closest GPS point
        for i, row in sync_df.iterrows():
            ts = row['master_timestamp']
            
            # Find closest GPS timestamp
            closest_idx = (gps_df['timestamp'] - ts).abs().idxmin()
            
            # Add GPS and map data to the synchronized dataframe
            sync_df.at[i, 'x'] = gps_df.at[closest_idx, 'x']
            sync_df.at[i, 'y'] = gps_df.at[closest_idx, 'y']
            
            # Find the nearest road segment (edge) from OSM
            # This is computationally expensive, so we'll do a simplified version
            # In a real implementation, use spatial indexing for efficiency
            
            # Get current position
            pos_x, pos_y = gps_df.at[closest_idx, 'x'], gps_df.at[closest_idx, 'y']
            
            # Find the nearest node in OSM
            nodes = osm_data['nodes']
            edges = osm_data['edges']
            
            # Calculate distances to all nodes (simplified)
            nodes['dist'] = np.sqrt((nodes.geometry.x - pos_x)**2 + (nodes.geometry.y - pos_y)**2)
            nearest_node = nodes.loc[nodes['dist'].idxmin()]
            
            # Get incident edges
            incident_edges = edges[edges['u'] == nearest_node.name]
            
            # Find road properties (simplified)
            road_type = "unknown"
            speed_limit = None
            lane_count = 1
            
            if len(incident_edges) > 0:
                edge = incident_edges.iloc[0]
                road_type = edge.get('highway', 'unknown')
                speed_limit = edge.get('maxspeed', None)
                lane_count = edge.get('lanes', 1)
            
            # Store metadata
            sync_df.at[i, 'road_type'] = road_type
            sync_df.at[i, 'speed_limit'] = speed_limit
            sync_df.at[i, 'lane_count'] = lane_count
        
        return sync_df
    
    def save_processed_data(self, sequence_id, sync_df, osm_data):
        """
        Save processed and aligned data
        
        Args:
            sequence_id (str): Sequence identifier
            sync_df (pd.DataFrame): Synchronized data
            osm_data (dict): OpenStreetMap data
            
        Returns:
            str: Path to saved data
        """
        # Create sequence directory
        seq_dir = os.path.join(self.output_dir, sequence_id)
        os.makedirs(seq_dir, exist_ok=True)
        
        # Create subdirectories
        seq_img_dir = os.path.join(seq_dir, 'images')
        seq_lidar_dir = os.path.join(seq_dir, 'lidar')
        seq_meta_dir = os.path.join(seq_dir, 'metadata')
        
        for dir_path in [seq_img_dir, seq_lidar_dir, seq_meta_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Process and save each frame
        for i, row in tqdm(sync_df.iterrows(), total=len(sync_df), desc=f"Processing sequence {sequence_id}"):
            frame_id = f"{i:06d}"
            
            # Process and save image
            if pd.notna(row['camera_file']) and os.path.exists(row['camera_file']):
                processed_img = self.process_image(row['camera_file'])
                if processed_img is not None:
                    cv2.imwrite(
                        os.path.join(seq_img_dir, f"{frame_id}.jpg"), 
                        cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                    )
            
            # Process and save LiDAR
            if pd.notna(row['lidar_file']) and os.path.exists(row['lidar_file']):
                processed_lidar = self.process_lidar(row['lidar_file'])
                if processed_lidar is not None:
                    # Save as numpy array for easier loading later
                    np.save(os.path.join(seq_lidar_dir, f"{frame_id}.npy"), processed_lidar)
            
            # Save metadata
            meta = {k: v for k, v in row.items() if k not in ['camera_file', 'lidar_file']}
            pd.Series(meta).to_json(os.path.join(seq_meta_dir, f"{frame_id}.json"))
        
        # Save the complete sync dataframe
        sync_df.to_csv(os.path.join(seq_dir, 'synchronized_data.csv'), index=False)
        
        # Save OSM data if available
        if osm_data is not None:
            os.makedirs(os.path.join(seq_dir, 'map'), exist_ok=True)
            osm_data['nodes'].to_file(os.path.join(seq_dir, 'map', 'nodes.geojson'), driver='GeoJSON')
            osm_data['edges'].to_file(os.path.join(seq_dir, 'map', 'edges.geojson'), driver='GeoJSON')
        
        return seq_dir
    
    def process_sequence(self, sequence_id):
        """
        Process a complete driving sequence
        
        Args:
            sequence_id (str): Sequence identifier
            
        Returns:
            str: Path to processed sequence
        """
        print(f"Processing sequence: {sequence_id}")
        
        # Find all data for this sequence
        sequence_data = self.find_sequence_data(sequence_id)
        
        # Extract timestamps
        raw_timestamps = self.extract_timestamps(sequence_data)
        
        # Interpolate timestamps to align sensors
        sync_df = self.interpolate_timestamps(raw_timestamps)
        
        # Fetch OSM data if GPS available
        osm_data = None
        if os.path.exists(sequence_data['gps']):
            osm_data = self.fetch_osm_data(sequence_data['gps'])
            
            # Align map data with sensor data
            if osm_data is not None:
                sync_df = self.align_map_data(osm_data, sync_df)
        
        # Save processed data
        processed_path = self.save_processed_data(sequence_id, sync_df, osm_data)
        
        print(f"Sequence {sequence_id} processed and saved to {processed_path}")
        return processed_path
    
    def process_all_sequences(self):
        """
        Process all sequences in the data root
        """
        # Find all sequence directories
        sequence_dirs = [d for d in os.listdir(self.data_root) 
                         if os.path.isdir(os.path.join(self.data_root, d))]
        
        for seq_id in sequence_dirs:
            self.process_sequence(seq_id)
        
        print(f"Processed {len(sequence_dirs)} sequences")


# Example usage
if __name__ == "__main__":
    # Set paths
    data_root = "/path/to/raw/data"
    output_dir = "/path/to/processed/dataset"
    
    # Initialize and run the dataset preparation
    preparator = DatasetPreparator(data_root, output_dir)
    
    # Option 1: Process all sequences
    preparator.process_all_sequences()
    
    # Option 2: Process a specific sequence
    # preparator.process_sequence("sequence_001")