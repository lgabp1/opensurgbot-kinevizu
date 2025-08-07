# opensurgbot-kinevizu

Submodule of [opensurgbot](https://github.com/lgabp1/opensurgbot).
Kinematic model and kinematic visualization for the da Vinci Large Needle Driver (Model No. 420006).

# Repository content

* [kinematics_study/KinevisuReport.pdf](./kinematics_study/KinevisuReport.pdf) provides some details about the kinematic model used
* [visualization/](./visualization/) provides a python-based 3D visualization of the proposed model

## KinevisuReport

A short report on the kinematic model.

## Visualization

A basic 3D matplotlib application to help verifying the model.

### Requirements

* [python 3.7+](https://python.org/)
* [numpy](https://numpy.org/) compatible with given python version
* [matplotlib](https://matplotlib.org/) compatible with given python version
* [pillow](https://github.com/python-pillow/Pillow) compatible with given python version

### Usage

Simply run the provided script with python `python kinevisu.py`, or import the `DeehliViz` class in an external script.

# License

MIT License, see [LICENSE](./LICENSE) file for details.