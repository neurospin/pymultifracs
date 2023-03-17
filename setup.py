import os.path as op
from setuptools import setup

with open('README.rst', 'r') as fid:
    long_description = fid.read()

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(op.join('pymultifracs', '_version.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


setup(name='pymultifracs',
      version=version,
      description='Implements wavelet based fractal and multifractal analysis '
                  'of 1d signals.',
      url='https://github.com/neurospin/pymultifracs',
      author='Omar Darwiche Domingues, Merlin Dumeur',
      author_email='',
      license='MIT',
      packages=['pymultifracs'],
      install_requires=[
          'numpy', 'scipy', 'scikit-learn', 'pywavelets', 'seaborn',
          'recombinator', 'statsmodels',
      ],
      zip_safe=False,
      python_requires='>=3.7',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved',
                   'Programming Language :: Python',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 3'],
      project_urls={
        'Documentation': 'https://neurospin.github.io/pymultifracs/',
        'Source': 'https://github.com/neurospin/pymultifracs',
        'Tracker': 'https://github.com/neurospin/pymultifracs/issues/',
      },
      platforms='any',
      )
