# Contributing to AtmPy

The ``atmpy`` package provides a set of tools for processing and analyzing data relevant to atmospheric science applications.  The package is written using Python 3.


## Contributing New Code

The field of atmospheric science is broad and developing and it is unlikely that this package will cover all use cases.  In that instance, a developer may reasonably desire to provide a tool that they use on a regular basis and think others may also find useful.

### Structure

Some effort has been placed in attempting to codify a standard directory structure.  The initial structure encompasses areas that those early collaborators found relevant to their own field of study.

```
. atmpy
|--__general
|--__aerosol
     |--__physics
          |--__mech
          |--__opt
     |--__instr
          |--__dma
          |--__opc
          |--__pas
          |--__crd
          |--__cpc
     |--__size_dstr
|--__radiation
|--__cloud_microphysics
```

If a developer is interested in providing new code, they will likely begin with the question where exactly does that code belong?  This question is a difficult one and while some thought has been placed in the initial package structure, there are many instances where the new code may not fit neatly into an existing module.  In that case

### Commenting

Documentation of files and modules should utilize [reStructuredText](http://www.sphinx-doc.org/en/stable/rest.html).

### Unit Tests

In order to maintain the integrity of the package, each new module or changes to existing modules should be tested extensively.  This is important as these modules may be used by a wide array of scientists and engineers and erroneous results may have significant impacts on the results of their work.

Code used for testing should *not* reside in the module directory itself, but rather should be contained in a directory within the existing module called ``/tests``.

## Continuous Integration
