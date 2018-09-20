from setuptools import setup

setup(
   name='cddd',
   version='0.1',
   description='continous and data-driven molecular descriptors (CDDD)',
   author='Robin Winter',
   author_email='robin.winter@bayer.com',
   packages=['cddd'],  #same as name
   package_data={'cddd': ['data/*', 'default_model/*']},
   include_package_data=True,
   #install_requires=['tensorflow'], #external packages as dependencies
)
