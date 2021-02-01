=======================
Installing pymultifracs
=======================

There are two ways to install this package: either by using a package manager to install the package only, which will make
the code only usable as an import,
or by cloning the repository first, and then installing the package which will make it editable

Installing the package only
===========================

Using conda
-----------

Creating a new environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

    wget https://raw.githubusercontent.com/neurospin/pymultifracs/master/env.yml
    conda env create -f env.yml

Installing into a pre-existing environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this package requires a recent version of python (>=3.7)

.. code:: shell

    wget https://raw.githubusercontent.com/neurospin/pymultifracs/master/env.yml
    conda env update -f env.yml --name $ENVNAME

----

Using pip
---------

.. code:: shell

    pip install git+https://github.com/neurospin/pymultifracs.git@master


Cloning the whole repository (including examples)
=================================================

Using conda
-----------

Using conda, the simplest way to proceed is to use the :code:`meta.yml` file to create or update
an environment, and then install the editable local version of pymultifracs on top.

Creating a new environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

    git clone https://github.com/neurospin/pymultifracs.git@master
    conda env create -f pymultifracs/meta.yml
    pip install -e pymultifracs

Installing into a pre-existing environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this package requires a recent version of python (>=3.7)

.. code:: shell

    git clone https://github.com/neurospin/pymultifracs.git@master
    conda env update -f pymultifracs/meta.yml --name $ENVNAME
    pip install -e pymultifracs

----

Using pip
---------

.. code:: shell

    git clone https://github.com/neurospin/pymultifracs.git@master
    pip install -e pymultifracs

See the examples/ folder, mainly examples/Demo.ipynb and examples/Maquette.ipynb
