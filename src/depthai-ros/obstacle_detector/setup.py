# setup.py
from setuptools import setup

package_name = 'obstacle_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hamizi',
    maintainer_email='your_email@example.com',
    description='Obstacle detection node using depth camera',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_detector = obstacle_detector.obstacle_detector:main'
        ],
    },
)