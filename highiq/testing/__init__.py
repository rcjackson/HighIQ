"""
This module contains test settings and files that are useful for
unit testing HighIQ.

You can test the processing from a sample ACF file by loading
:func:`highiq.testing.TEST_FILE`.
"""

import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
TEST_FILE = os.path.join(DATA_PATH, 'testsig.a1.20180801.000003.nc')