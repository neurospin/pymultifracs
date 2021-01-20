from setuptools import setup

setup(name='pymultifracs',
      version='0.1',
      description='Implements wavelet based fractal and multifractal analysis '
                  'of 1d signals.',
      url='https://github.com/neurospin/pymultifracs',
      author='Omar Darwiche Domingues, Merlin Dumeur',
      author_email='',
      license='',
      packages=['pymultifracs'],
      install_requires=[
          'numpy', 'scipy', 'scikit-learn', 'pywavelets', 'seaborn'
      ],
      zip_safe=False,
      python_requires='>=3.7')
