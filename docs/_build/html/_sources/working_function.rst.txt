.. _working_function:

Working function
===================

This section contains the working functions of the **dagrad** library. These functions are used to learn the structure of the DAG model.
It includes the implementation of the NOTEARS, DAGMA, and TOPO methods for linear and nonlinear model using numpy and torch. It includes the details of the options that can be specified for the methods.


Notears
^^^^^^^
.. autofunction:: dagrad.method.notears.notears_linear_numpy
.. autofunction:: dagrad.method.notears.notears_linear_torch
.. autofunction:: dagrad.method.notears.notears_nonlinear

Dagma
^^^^^^^^

.. autofunction:: dagrad.method.dagma.dagma_linear_numpy
.. autofunction:: dagrad.method.dagma.dagma_linear_torch
.. autofunction:: dagrad.method.dagma.dagma_nonlinear

Topo
^^^^

.. autofunction:: dagrad.method.topo.topo_linear
.. autofunction:: dagrad.method.topo.topo_nonlinear