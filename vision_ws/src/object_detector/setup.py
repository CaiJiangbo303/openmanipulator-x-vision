from setuptools import find_packages, setup

package_name = 'object_detector'

setup(
    name=package_name,
    version='0.1.0',

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
        (
            'share/' + package_name + '/config',
            ['config/detector.yaml']
        ),
    ],

    install_requires=[
        'setuptools',
    ],

    zip_safe=True,

    maintainer='CaiJiangbo303',

    description=(
        'Pluggable object detector for '
        'OpenMANIPULATOR-X visual grasping.'
    ),

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'object_detector = '
            'object_detector.detector_node:main',
        ],
    },
)
