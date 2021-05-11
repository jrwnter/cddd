from setuptools import setup

setup(
    name='cddd',
    version='1.1',
    packages=['cddd', 'cddd.data'],
    package_data={'cddd': ['data/*', 'data/default_model/']},
    include_package_data=True,
    url='https://github.com/jrwnter/cddd',
    download_url='https://github.com/jrwnter/cddd/archive/refs/tags/1.1.tar.gz',
    license='MIT',
    author='Robin Winter',
    author_email='robin.winter@bayer.com',
    description='continous and data-driven molecular descriptors (CDDD)',
    python_requires='>=3.6.1, <3.7',
    install_requires=[
        'tensorflow-gpu==1.10.0',
        'scikit-learn',
        'pandas<=1.0.3',
        'requests',
        'appdirs'
      ],
    extras_require = {
        'cpu' : [
            'tensorflow==1.10.0'
            ]
    },
    entry_points={
        'console_scripts': [
            'cddd = cddd.run_cddd:main_wrapper',
        ],
    },
)
