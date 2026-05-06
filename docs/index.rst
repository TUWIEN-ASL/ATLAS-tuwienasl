.. atlas_gui documentation master file, created by
   sphinx-quickstart on Wed Jun 11 14:23:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ATLAS Documentation
===================

**ATLAS** is an annotation tool for long-horizon robotic action segmentation.
It provides time-synchronized visualization of multi-modal robotic data,
including multi-view video, ROS bags, and proprioceptive signals, and
supports annotation of action boundaries, action labels, and task outcomes.

The keyboard-centric interface keeps annotation effort low, and a modular
dataset abstraction layer makes it straightforward to extend ATLAS to new
formats. Out of the box, the tool supports **REASSEMBLE**, **RLDS**,
**ROS bags**, generic **video**, and **image sequences**.

.. tip::
   * **Source code:** `github.com/TUWIEN-ASL/ATLAS-tuwienasl <https://github.com/TUWIEN-ASL/ATLAS-tuwienasl>`_
   * **Paper:** `arXiv:2604.26637 <https://arxiv.org/abs/2604.26637>`_
   * **Issues / feature requests:** `GitHub Issues <https://github.com/TUWIEN-ASL/ATLAS-tuwienasl/issues>`_

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   configuration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
