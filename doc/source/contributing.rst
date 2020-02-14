==========================
Contributor's Guide
==========================

One pesky thing about Impostor's Syndrome is that it tells you that you are
not good enough to make a contribution. Let us assure you that that is very
much false. We welcome contributions from anyone no matter what their skill
level is. Contributions can be as easy as improving the documentation to as
complex as adding new functionality for different instruments.

There are just some things that we ask of you. One is that your code be able
to be distributed under the BSD 3-clause license, which is available in LICENSE
in the main directory.

One, we ask, that when on the GitHub forum or making contributions to HighIQ
that all developers and users follow the HighIQ code of conduct.


Contributor Covenant Code of Conduct
------------------------------------
**Our Pledge**

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project
and our community a harassment-free experience for everyone, regardless of
age, body size, disability, ethnicity, gender identity and expression,
level of experience, nationality, personal appearance, race, religion,
or sexual identity and orientation.

**Our Standards**

Examples of behavior that contributes to creating a positive environment include:

    Using welcoming and inclusive language

    Being respectful of differing viewpoints and experiences

    Gracefully accepting constructive criticism

    Focusing on what is best for the community

    Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

    The use of sexualized language or imagery and unwelcome sexual attention or
    advances

    Trolling, insulting/derogatory comments, and personal or political attacks

    Public or private harassment

    Publishing others' private information, such as a physical or electronic
    address, without explicit permission

    Other conduct which could reasonably be considered inappropriate in a
    professional setting

**Our Responsibilities**

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

**Scope**

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an
appointed representative at an online or offline event. Representation of a
project may be further defined and clarified by project maintainers.

**Enforcement**

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project team at
`rjackson@anl.gov <mailto:rjackson@anl.gov>`_. The project team will review
and investigate all complaints, and will respond in a way that it deems
appropriate to the circumstances. The project team is obligated to maintain
confidentiality with regard to the reporter of an incident. Further details
of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in
good faith may face temporary or permanent repercussions as determined
by other members of the project's leadership.

**Attribution**

This Code of Conduct is adapted from the Contributor Covenant, version 1.4,
available at `<http://contributor-covenant.org/version/1/4>`_

Code Style
----------

HighIQ follows the PEP8 code standards. To make sure the code follows the PEP8
style, there are checkers available out there such as pylint and pycodestyle.

For more on PEP8 style:
    `<https://www.python.org/dev/peps/pep-0008/>`_

To install pycode style:
::
    conda install pycodestyle
::

To install pylint:
::
    conda install pylint
::


Python File Setup
-----------------

In a new .py file, the top of the code should have the function, sphinx comments
and the public and private functions within the .py file. Public functions are
listed first and then private functions and classes. Private functions should
have an underscore in front of the name. A space is needed between the last
function and the closing docstring quotation marks.

Following the introduction code, modules are then added. To follow PEP8
standards, modules should be added in the following order:

    1. Standard library imports
    2. Related third party imports
    3. Local application imports

Following the main function def line, but before the code within it, a docstring is
needed to explain all arguments, returns, references, and other information. Please
follow the NumPy documentation style:

`<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_

For an example format of the documentation, see this:

.. code-block:: python

    def get_lidar_moments(spectra, snr_thresh=0, block_size_ratio=1.0, which_moments=None):
        """
        This function will retrieve the lidar moments of the Doppler spectra.

        Parameters
        ----------
        spectra: ACT Dataset
            The dataset containing the processed Doppler spectral density functions.
        snr_thresh: float
            The minimum signal to noise ratio to use as an initial mask of noise.
        block_size_ratio: float
            This value is used to determine how much data the GPU will process in one loop. If your
            GPU has more memory, you may be able to optimize processing by raising this number. In
            addition, if you encounter out of memory errors, try lowering this number, ensuring that
            it is a positive floating point number.
        which_moments: list or None
            This tells HighIQ which moments should be processed. If this list is None, then the
            signal to noise ratio, doppler velocity, spectral width, skewness,
            and kurtosis will be calculated.

        Returns
        -------
        spectra:
            The database with the Doppler lidar moments.
        """

        (your code is here)


Testing
-------

When adding a new function to HighIQ it is important to add it to the __init__.py
under the corresponding folder.

Create a test function and use assert to test the calculated values against known
values. For an example, see:

`<https://github.com/rcjackson/HighIQ/blob/master/tests/test_highiq.py>`_

Pytest will run this test whenever a pull request is made to the master branch
of the openradar/HighIQ repository. This will then allow the maintainers to
determine how the pull request will affect the functionality of HighIQ.


.. code-block:: python

    def test_io():
        my_ds = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)
        assert 'acf' in my_ds.variables.keys()
        assert 'acf_bkg' in my_ds.variables.keys()
        my_ds.close()

GitHub
------

When you make contributions to HighIQ, we ask that you make your own fork
of openradar/HighIQ and create your own branch from within that fork. After
forking the repository on GitHub, create your own branch by doing:

::

   git checkout -b this_branch
   git branch this_branch

::

Make your changes, commit, and then to push to that branch do:
::
   git push origin this_branch
::

After that is done, make a pull request from that branch to the master branch
on openradar/HighIQ where the maintainers will review your pull request.
