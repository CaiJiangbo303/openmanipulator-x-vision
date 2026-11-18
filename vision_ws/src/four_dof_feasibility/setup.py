from setuptools import find_packages, setup

package_name = 'four_dof_feasibility'

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
        'Four-DOF feasibility solver for '
        'the inverted OpenMANIPULATOR-X.'
    ),

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            (
                'four_dof_feasibility = '
                'four_dof_feasibility.feasibility_node:main'
            ),
        ],
    },
)
