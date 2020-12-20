from setuptools import setup, find_packages

setup(
    name='finite-mdp',
    version='1.0.dev0',
    description='Gym environment for MDPs with finite state and action spaces',
    url='https://github.com/eleurent/finite-mdp',
    author='Edouard Leurent',
    author_email='eleurent@gmail.com',
    classifiers=[
        'Intended Audience :: Researchers',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='finite mdp',
    packages=find_packages(exclude=['docs', 'scripts', 'tests*']),
    install_requires=['gym', 'numpy', 'matplotlib', 'torch>=1.2.0', 'networkx'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
    },
    entry_points={
        'console_scripts': [],
    },
)
