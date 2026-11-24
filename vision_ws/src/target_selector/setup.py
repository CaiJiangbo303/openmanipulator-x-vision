from setuptools import find_packages, setup

package_name = 'target_selector'

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

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='bo',
    maintainer_email='bo@todo.todo',

    description=(
        'Dynamic grasp target selector.'
    ),

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'state_gate = target_selector.state_gate_node:main',
            (
                'target_selector = '
                'target_selector.selector_node:main'
            ),
        ],
    },
)
