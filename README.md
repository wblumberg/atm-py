# atm-py
Set of python packages for processing atmospheric aerosol data.

## Roadmap

Currently, nothing is fixed in stone.  The initial check-in of python packages shows a possible outline of the repository setup, but this is flexible.  The current packages may be further broken out and definitely expanded.  As they stand now, they are

* **constants.py** - This file simply contains physical constants that are commonly used by many calculations and may be imported for calculations not related to the current set of functions in the packages defined below.
* **atmosphere.py** - This package contains functions that may be used in aerosol physics calculations but also may be applied in instances not used here.  These include calculations of viscocity, mean free path, etc.  Many of the calculations may be expanded to include gas types or different conditions.
* **aerosol.py** - these are aerosol specific calculations that concern different properties of particles.  These calculations may be used in various data analysis situations.  This file may be especially prone to change and further subdividing.  For instance, there may be calculations that are specific only to a particular instrument.

## Documentation

At this time, there is no defined documentation standard.  At this point, the docstring is used to provide a short summary while the @ notation is used to define parameters, outputs and sources.

## Dependences
python ?

numpy
scipy
pandas
matplotlib