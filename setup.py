from setuptools import setup

setup(
    name='cddd',
    version='0.1',
    packages=['cddd', 'cddd.data'],
    package_data={'cddd': ['data/*', 'data/default_model/']},
    include_package_data=True,
    url='',
    license='',
    author='Robin Winter',
    author_email='robin.winter@bayer.com',
    description='continous and data-driven molecular descriptors (CDDD)',
    entry_points={
        'console_scripts': [
            'cddd = cddd.run_cddd:main_wrapper',
        ],
    },
)
