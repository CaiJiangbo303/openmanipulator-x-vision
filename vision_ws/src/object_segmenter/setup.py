from setuptools import find_packages, setup

package_name = 'object_segmenter'

setup(
    name=package_name,
    version='0.1.0',

    packages=find_packages(exclude=['test']),

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

    maintainer='CaiJiangbo303',
    maintainer_email='236074124+CaiJiangbo303@users.noreply.github.com',

    description='SAM2 instance segmentation for OpenMANIPULATOR-X visual grasping.',
    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'object_segmenter = object_segmenter.launcher:main',
        ],
    },
)
