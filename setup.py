from setuptools import setup

setup(name='mfanalysis',
      version='0.12dev',
      description='Implements wavelet based multifractal analysis of 1d signals.',
      url='https://gitlab.inria.fr/odarwich/omar_darwiche_domingues',
      author='Omar Darwiche Domingues',
      author_email='',
      license='',
      packages=['mfanalysis'],
      install_requires=[
          'pywavelets', 'scipy', 'numpy', 'matplotlib',
      ],
      zip_safe=False)
