from setuptools import setup

setup(name='mfanalysis',
      version='0.13dev',
      description='Implements wavelet based multifractal analysis of 1d signals.',
      url='https://github.com/neurospin/mfanalysis',
      author='Omar Darwiche Domingues',
      author_email='',
      license='',
      packages=['mfanalysis'],
      install_requires=[
          'pywavelets', 'scipy', 'numpy', 'matplotlib',
      ],
      zip_safe=False)
