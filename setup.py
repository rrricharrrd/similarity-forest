from setuptools import setup


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
      install_requires=['numpy'],
      scripts=[],
      zip_safe=False)
