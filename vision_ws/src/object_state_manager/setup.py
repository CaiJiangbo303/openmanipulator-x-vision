from setuptools import find_packages, setup

package_name = 'object_state_manager'

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
        'setuptools',
    ],

    zip_safe=True,

    maintainer='bo',
    maintainer_email='bo@todo.todo',

    description=(
        'Persistent object state manager '
        'for RGB-D perception.'
    ),

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            (
                'object_state_manager = '
                'object_state_manager.state_manager_node:main'
            ),
        ],
    },
)
