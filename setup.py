import os
from setuptools import setup, find_packages
import versioneer
import sys


# https://www.pydanny.com/python-dot-py-tricks.html
if sys.argv[-1] == 'test':
    test_requirements = [
        'pytest',
        'coverage',
        'pytest_cov',
    ]
    try:
        modules = map(__import__, test_requirements)
    except ImportError as e:
        err_msg = e.message.replace("No module named ", "")
        msg = "%s is not installed. Install your test requirments." % err_msg
        raise ImportError(msg)
    r = os.system('py.test test -v --cov=csirtg_domainsml_tf --cov-fail-under=60')
    if r == 0:
        sys.exit()
    else:
        raise RuntimeError('tests failed')


data_files = [
    'data/model.h5',
    'data/word-dict.json',
    'data/weights.h5'
]

setup(
    name="csirtg_domainsml_tf",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="CSIRTG DOMAINs ML Framework - TensorFlow",
    long_description="",
    url="https://github.com/csirtgadgets/csirtg-domainsml-tf-py",
    license='MPLv2',
    data_files=[(os.path.join('csirtg_domainsml_tf', 'data'), data_files)],
    keywords=['network', 'security'],
    author="Wes Young",
    author_email="wes@csirtgadgets.com",
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'pandas',
        'keras'
    ],
    entry_points={
       'console_scripts': [
           'csirtg-domainsml-tf-train=csirtg_domainsml_tf.train:main',
           'csirtg-domainsml-tf=csirtg_domainsml_tf:main'
       ]
    },
)
