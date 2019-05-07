from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pandas', 'numpy', 'keras', 'scikit-learn', 'h5py', 'pyEDFlib', 'tensorflow']

setup(name='trainer',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      include_package_data=True,
      description='Trainer setup for Sleep Model',
      zip_safe=False,
      author='Rahul Shah',
      author_email='rshah98626@gmail.com')
