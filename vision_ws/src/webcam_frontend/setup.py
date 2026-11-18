from setuptools import setup

package_name = 'webcam_frontend'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bo',
    maintainer_email='bo@localhost',
    description='Windows webcam TCP frontend for ROS2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'webcam_frontend = '
            'webcam_frontend.webcam_frontend_node:main',
        ],
    },
)
