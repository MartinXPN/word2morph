from setuptools import find_packages, setup

setup(
    name='word2morph',
    version='0.2.0',
    description='Python package for neural morpheme extraction from words',
    author='Martin Mirakyan',
    author_email='mirakyanmartin@gmail.com',
    python_requires='>=3.6.0',
    url='https://github.com/MartinXPN/word2morph',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'GitPython>=2.1.11',
        'matplotlib>=3.0.3',
        'baytune>=0.2.4',
        'fire>=0.1.3',
        'joblib>=0.13.2',
        'jupyter-core>=4.4.0',
        'Keras>=2.2.4',
        'numpy>=1.16.1',
        'scikit-learn>=0.20.2',
        'tensorflow>=1.12.0',
        'tqdm>=4.31.1',
        'keras-contrib @ git+https://www.github.com/keras-team/keras-contrib.git',
    ],
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
