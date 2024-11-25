from setuptools import setup, find_packages

package_name = 'obstacle_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),  # Changed this line
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy',
        'opencv-python-headless',
    ],
    zip_safe=True,
    maintainer='hamizi',
    maintainer_email='your_email@example.com',
    description='Obstacle detection node using depth camera',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_detector = obstacle_detector.obstacle_detector_2d:main',
            'cloud_detector = obstacle_detector.point_cloud_detector:main'
        ],
    },
    package_data={'': ['resource/*']},
    python_requires='>=3.6',
)