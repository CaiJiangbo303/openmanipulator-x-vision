from setuptools import find_packages, setup

package_name = 'object_motion_predictor'

setup(
    name=package_name,
    version='0.0.1',

    packages=find_packages(),

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
    ],

    zip_safe=True,

    maintainer='bo',
    maintainer_email='bo@todo.todo',

    description=(
        'Constant-velocity object motion predictor.'
    ),

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            (
                'object_motion_predictor = '
                'object_motion_predictor.predictor_node:main'
            ),
        ],
    },
)
