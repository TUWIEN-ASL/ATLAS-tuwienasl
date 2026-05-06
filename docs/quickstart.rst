Quick Start
===========

This page walks you through installing ATLAS and running your first annotation
session. For a deeper look at how to configure ATLAS for your own data, see
:doc:`configuration`.

Prerequisites
-------------

* **Python** ≥ 3.10
* **conda** (recommended for environment isolation)

Installation
------------

Clone the repository and install ATLAS in editable mode inside a fresh
conda environment:

.. code-block:: bash

   # Clone the repository
   git clone git@github.com:TUWIEN-ASL/ATLAS-tuwienasl.git
   cd ATLAS-tuwienasl

   # Create and activate a conda environment
   conda create -n gui python=3.10
   conda activate gui

   # Install ATLAS
   pip install -e .

To additionally install the dependencies needed to build this documentation
locally, run:

.. code-block:: bash

   pip install -e ".[docs]"

Running the GUI
---------------

Launch the annotation tool by passing one of the example configuration files
to the GUI module:

.. code-block:: bash

   python -m atlas_gui.gui --config config/rlds.yaml

The :gh-tree:`config/ <config>` folder contains ready-to-use configurations for
each supported dataset format (REASSEMBLE, RLDS, ROS bags, video, frames).
Pick the one that matches your data, copy it, and edit it to point at your own
files.

.. important::
   Annotations made on a ``REASSEMBLE`` dataset with
   ``annotation_storage: h5`` are written **in place**, overwriting the
   original HDF5 file. Use ``annotation_storage: json`` if you would prefer
   ATLAS to write annotations to an external file instead.

Default keyboard shortcuts
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the GUI is running, the following keys control playback and annotation
out of the box:

================  =============================================
Key               Action
================  =============================================
**S**             Start / stop annotating an action
**Q** / **E**     Load previous / next segment
**A** / **D**     Step backwards / forwards by ``ff_value_big``
**Z** / **C**     Step backwards / forwards by ``ff_value_small``
**F**             Jump to the end of the current segment
**Space**         Play / pause
**Backspace**     Delete the most recent annotation
**P**             Save annotations for the current segment
================  =============================================

All shortcuts are configurable — see :doc:`configuration` for details.

Next steps
----------

* :doc:`configuration` — full reference for every configuration key, with
  per-format examples.
* :doc:`modules` — auto-generated API reference for developers extending the
  tool with new dataset types.
