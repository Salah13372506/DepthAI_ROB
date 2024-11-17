#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class DepthToGrayConverter(Node):
    def __init__(self):
        super().__init__('depth_to_gray')
        
        self.subscription = self.create_subscription(
            Image,
            '/stereo/depth',
            self.depth_callback,
            10)
            
        self.publisher = self.create_publisher(
            Image,
            '/stereo/depth_gray',
            10)
            
        self.bridge = CvBridge()
        
    def depth_callback(self, msg):
        try:
            # Conversion de l'image de profondeur
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            # Remplacer les valeurs invalides (inf ou nan) par 0
            depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Définition des seuils de profondeur (en mètres)
            min_depth = 0.2  # 20cm
            max_depth = 5.0  # 5m
            
            # Application d'un masque pour les pixels valides
            mask = (depth_image > min_depth) & (depth_image < max_depth)
            
            # Normalisation avec amélioration du contraste
            depth_normalized = np.zeros_like(depth_image)
            depth_normalized[mask] = depth_image[mask]
            
            # Conversion en uint8 avec étirement d'histogramme
            depth_gray = cv2.normalize(depth_normalized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Option: Amélioration du contraste avec égalisation d'histogramme
            depth_gray = cv2.equalizeHist(depth_gray)
            
            # Option: Appliquer un filtre de lissage pour réduire le bruit
            depth_gray = cv2.GaussianBlur(depth_gray, (3,3), 0)
            
            # Création du message Image ROS2
            gray_msg = self.bridge.cv2_to_imgmsg(depth_gray, encoding='mono8')
            gray_msg.header = msg.header
            
            # Publication de l'image
            self.publisher.publish(gray_msg)
            
        except Exception as e:
            self.get_logger().error(f'Erreur: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DepthToGrayConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()