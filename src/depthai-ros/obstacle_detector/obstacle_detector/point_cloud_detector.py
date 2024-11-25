#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from sklearn.cluster import DBSCAN
import struct
import sensor_msgs_py.point_cloud2 as pc2

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        
        # Parameters
        self.declare_parameter('min_height', 0.2)  # Min height from ground
        self.declare_parameter('max_height', 2.0)  # Max height to consider
        self.declare_parameter('min_distance', 0.5)  # Minimum distance to consider obstacle
        self.declare_parameter('max_distance', 5.0)  # Maximum distance to consider
        self.declare_parameter('cluster_tolerance', 0.3)  # DBSCAN eps parameter
        self.declare_parameter('min_cluster_size', 100)  # Minimum points to form obstacle
        
        # Publishers and Subscribers
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            '/oak_d_pro/points',  # Adjust topic name as needed
            self.point_cloud_callback,
            10)
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            'detected_obstacles',
            10)
            
    def point_cloud_callback(self, cloud_msg):
        # Convert PointCloud2 to numpy array
        points = np.array(list(pc2.read_points(cloud_msg, 
                                             field_names=("x", "y", "z"),
                                             skip_nans=True)))
        
        # Filter points by height and distance
        min_height = self.get_parameter('min_height').value
        max_height = self.get_parameter('max_height').value
        min_distance = self.get_parameter('min_distance').value
        max_distance = self.get_parameter('max_distance').value
        
        mask = np.logical_and.reduce((
            points[:, 2] >= min_height,
            points[:, 2] <= max_height,
            np.linalg.norm(points[:, :2], axis=1) >= min_distance,
            np.linalg.norm(points[:, :2], axis=1) <= max_distance
        ))
        
        filtered_points = points[mask]
        
        if len(filtered_points) < 10:
            return
            
        # Cluster points using DBSCAN
        eps = self.get_parameter('cluster_tolerance').value
        min_samples = self.get_parameter('min_cluster_size').value
        
        clusters = DBSCAN(eps=eps, 
                         min_samples=min_samples, 
                         algorithm='ball_tree').fit(filtered_points)
        
        # Create markers for visualization
        marker_array = MarkerArray()
        
        for cluster_id in range(max(clusters.labels_) + 1):
            cluster_points = filtered_points[clusters.labels_ == cluster_id]
            
            if len(cluster_points) < min_samples:
                continue
                
            # Calculate cluster centroid and dimensions
            centroid = np.mean(cluster_points, axis=0)
            dimensions = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
            
            # Create marker
            marker = Marker()
            marker.header = cloud_msg.header
            marker.ns = "obstacles"
            marker.id = cluster_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set position (centroid)
            marker.pose.position.x = centroid[0]
            marker.pose.position.y = centroid[1]
            marker.pose.position.z = centroid[2]
            
            # Set dimensions
            marker.scale.x = max(dimensions[0], 0.1)
            marker.scale.y = max(dimensions[1], 0.1)
            marker.scale.z = max(dimensions[2], 0.1)
            
            # Set color (red for obstacles)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker_array.markers.append(marker)
            
        # Publish markers
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = ObstacleDetector()
    rclpy.spin(obstacle_detector)
    obstacle_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()