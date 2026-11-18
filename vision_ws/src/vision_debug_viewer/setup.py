from setuptools import find_packages, setup

package_name = 'vision_debug_viewer'

setup(
    name=package_name,
    version='0.0.1',

    packages=find_packages(
        exclude=['test']
    ),

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

    install_requires=[
        'setuptools'
    ],

    zip_safe=True,

    maintainer='bo',
    maintainer_email='bo@localhost',

    description=(
        'RGB visual debug viewer for '
        'YOLO, tracker and SAM2'
    ),

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'viewer = '
            'vision_debug_viewer.viewer_node:main',
        ],
    },
)
