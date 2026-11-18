from setuptools import find_packages, setup

package_name = 'object_localizer'

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
    ],

    install_requires=[
        'setuptools',
        'numpy',
    ],

    zip_safe=True,

    maintainer='CaiJiangbo303',

    description=(
        'RGB-D 3D localization for '
        'OpenMANIPULATOR-X visual grasping.'
    ),

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'object_localizer = '
            'object_localizer.localizer_node:main',
            'mask_localizer = '
            'object_localizer.mask_localizer_node:main',
        ],
    },
)
