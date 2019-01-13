from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='',
      author='TH',
      author_email='thilo@colabel.io',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'Pillow',
          'interruptingcow'
      ],
      zip_safe=False)
