from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='Similarity Forest',
      version='0.1',
      description='Similarity Forest',
      url='https://github.com/rrricharrrd/similarity-forest',
      author='Richard Harris',
      author_email='rrricharrrd@gmail.com',
      license='MIT',
      packages=['simforest'],
      include_dirs=[np.get_include()],
      ext_modules=cythonize('simforest/_node.pyx'),
      scripts=[])
