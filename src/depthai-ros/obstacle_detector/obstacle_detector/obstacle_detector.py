#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension


#Added a comment to test the pushes

#Added a comment to test Vscode source control
class ObstacleDetectorNode(Node):
    def __init__(self):
        super().__init__('obstacle_detector_node')
        
        # Paramètres configurables
        self.declare_parameter('distance_threshold', 30)    
        self.declare_parameter('min_blob_size', 500)       
        self.declare_parameter('max_blob_size', 10000)     
        self.declare_parameter('blur_size', 5)             
        self.declare_parameter('bilateral_d', 5)           
        self.declare_parameter('bilateral_sigma', 50)      
        
        self.bridge = CvBridge()
        
        self.depth_sub = self.create_subscription(
            Image,
            '/stereo/converted_depth',
            self.depth_callback,
            10
        )
        
        self.obstacle_pub = self.create_publisher(
            Float32MultiArray,
            '/obstacles/positions',
            10
        )
        
        self.get_logger().info('Obstacle detector node initialized')
        self.first_frame = True

    def preprocess_depth_image(self, image):
        """Prétraitement de l'image de profondeur en gardant la cohérence des valeurs."""
        # Copie de l'image pour éviter de modifier l'original
        depth_copy = image.copy()
        
        # Remplacer les valeurs invalides par 255 (loin)
        depth_copy[~np.isfinite(depth_copy)] = 255
        
        # Application d'un filtre bilatéral pour préserver les bords
        bilateral_d = self.get_parameter('bilateral_d').value
        bilateral_sigma = self.get_parameter('bilateral_sigma').value
        filtered = cv2.bilateralFilter(depth_copy.astype(np.uint8), bilateral_d, bilateral_sigma, bilateral_sigma)
        
        # Application d'un filtre gaussien pour lisser davantage
        blur_size = self.get_parameter('blur_size').value
        if blur_size % 2 == 0:
            blur_size += 1
        filtered = cv2.GaussianBlur(filtered, (blur_size, blur_size), 0)
        
        return filtered

    def depth_callback(self, depth_msg):
        try:
            # Conversion en image OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            if self.first_frame:
                height, width = depth_image.shape
                self.get_logger().info(f'Image dimensions - Width: {width}, Height: {height}')
                self.first_frame = False
            
            # Prétraitement de l'image
            filtered_depth = self.preprocess_depth_image(depth_image)
            
            # Récupération des paramètres
            distance_threshold = self.get_parameter('distance_threshold').value
            min_blob_size = self.get_parameter('min_blob_size').value
            max_blob_size = self.get_parameter('max_blob_size').value
            
            # Création d'un masque pour les obstacles proches
            # Les pixels < threshold restent à 0 (obstacles proches)
            # Les pixels >= threshold deviennent 255 (zones éloignées)
            mask = cv2.threshold(filtered_depth, distance_threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Opérations morphologiques
            kernel = np.ones((7,7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Inverser le masque pour la détection des composantes connexes
            # car connectedComponents cherche les zones blanches
            mask_inv = cv2.bitwise_not(mask)
            
            # Détection des composantes connexes sur le masque inversé
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask_inv.astype(np.uint8), 
                connectivity=8
            )
            
            # Liste pour stocker les positions des obstacles
            obstacle_positions = []
            
            # Créer une image pour la visualisation
            visualization = cv2.cvtColor(filtered_depth, cv2.COLOR_GRAY2BGR)
            
            # Analyse de chaque composante connexe
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_blob_size <= area <= max_blob_size:
                    x = int(centroids[i][0])
                    y = int(centroids[i][1])
                    obstacle_mask = (labels == i)
                    distance = np.mean(depth_image[obstacle_mask])
                    
                    # Vérifier si c'est vraiment un obstacle proche
                    if distance < distance_threshold:
                        cv2.circle(visualization, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(visualization, f'D:{distance:.1f}', (x+10, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        obstacle_positions.extend([float(x), float(y), float(distance), float(area)])
            
            # Afficher les images
            cv2.imshow('Original Depth', depth_image)
            cv2.imshow('Filtered Depth', filtered_depth)
            cv2.imshow('Mask', mask)  # Les obstacles sont en noir
            cv2.imshow('Detection', visualization)
            cv2.waitKey(1)
            
            # Publication des résultats
            msg = Float32MultiArray()
            msg.data = obstacle_positions
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "obstacles"
            msg.layout.dim[0].size = len(obstacle_positions) // 4
            msg.layout.dim[0].stride = len(obstacle_positions)
            
            self.obstacle_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectorNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()