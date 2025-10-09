Installing PyMultifracs
-----------------------

Using pip
*********

.. code:: shell

    pip install pymultifracs

Optional features
+++++++++++++++++

In order to avoid download unnecessary packages, by default only the essential dependencies are installed. The options `robust` and `bootstrap` allow the use of the respective features in the toolbox:

.. code:: shell

    pip install pymultifracs[robust,bootstrap]

Alternatively, the "full" option installs all optional packages.

.. code:: shell

    pip install pymultifracs[full]


Cloning the whole repository (including examples)
*************************************************

An editable installation is possible by first cloning the repository, and then installing it as an editable package.

.. code:: shell

    git clone https://github.com/neurospin/pymultifracs
    pip install -e pymultifracs

Examples are found in the ``examples/`` folder.