#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class SimpleDepthObstacleDetector(Node):
    def __init__(self):
        super().__init__('simple_depth_obstacle_detector')
        
        self.bridge = CvBridge()
        
        # Paramètres ajustables
        self.declare_parameter('depth_threshold', 150)  # Augmenté pour mieux voir
        self.declare_parameter('min_area', 1000)  # Filtre les petits objets
        
        # Publishers supplémentaires pour debug
        self.depth_pub = self.create_publisher(Image, 'debug/depth', 10)
        self.binary_pub = self.create_publisher(Image, 'debug/binary', 10)
        self.viz_pub = self.create_publisher(Image, 'obstacle_visualization', 10)
        
        self.depth_sub = self.create_subscription(
            Image,
            '/stereo/converted_depth',
            self.depth_callback,
            10)
            
    def depth_callback(self, depth_msg):
        # Conversion et nettoyage de l'image
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalisation
        depth_min = np.min(depth_image)
        depth_max = np.max(depth_image)
        
        if depth_max - depth_min > 0:
            depth_normalized = ((depth_image - depth_min) * 255 / (depth_max - depth_min))
        else:
            depth_normalized = np.zeros_like(depth_image)
        
        depth_uint8 = np.clip(depth_normalized, 0, 255).astype(np.uint8)
        
        # Amélioration du contraste
        depth_uint8 = cv2.equalizeHist(depth_uint8)
        
        # Appliquer un filtre médian pour réduire le bruit
        depth_filtered = cv2.medianBlur(depth_uint8, 5)
        
        # Publier l'image de profondeur normalisée
        depth_colored = cv2.applyColorMap(depth_filtered, cv2.COLORMAP_JET)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_colored, encoding="bgr8")
        self.depth_pub.publish(depth_msg)
        
        # Seuillage
        threshold = self.get_parameter('depth_threshold').value
        _, binary = cv2.threshold(depth_filtered, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Nettoyage du masque binaire
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Publier l'image binaire
        binary_msg = self.bridge.cv2_to_imgmsg(binary, encoding="mono8")
        self.binary_pub.publish(binary_msg)
        
        # Trouver les contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Créer image de visualisation
        viz_image = cv2.cvtColor(depth_filtered, cv2.COLOR_GRAY2BGR)
        
        # Dessiner les contours et rectangles
        min_area = self.get_parameter('min_area').value
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Dessiner le contour en rouge
            cv2.drawContours(viz_image, [contour], -1, (0, 0, 255), 2)
            
            # Dessiner le rectangle en vert
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Ajouter le texte avec l'aire
            text = f'Area: {int(area)}'
            cv2.putText(viz_image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
        
        # Publier la visualisation
        viz_msg = self.bridge.cv2_to_imgmsg(viz_image, encoding="bgr8")
        viz_msg.header = depth_msg.header
        self.viz_pub.publish(viz_msg)

def main(args=None):
    rclpy.init(args=args)
    detector = SimpleDepthObstacleDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()