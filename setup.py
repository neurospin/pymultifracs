from setuptools import setup

setup(name='mfanalysis',
      version='0.20',
      description='Implements wavelet based fractal and multifractal analysis of 1d signals.',
      url='https://github.com/neurospin/mfanalysis',
      author='Omar Darwiche Domingues, Merlin Dumeur',
      author_email='',
      license='',
      packages=['mfanalysis'],
      install_requires=[
          'numpy', 'scipy', 'scikit-learn', 'pywavelets', 'seaborn'
      ],
      zip_safe=False,
      python_requires='>=3.7')
