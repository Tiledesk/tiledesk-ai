from setuptools import setup
import glob
import os

with open('requirements.txt') as f:
    required = [x for x in f.read().splitlines() if not x.startswith("#")]

from tileai.cli import __version__, _program

setup(name=_program,
      version=__version__,
      packages=['tileai','tileai.cli','tileai.core','tileai.shared'],
      description='tiledesk ai',
      url='https://github.com/Tiledesk/tiledesk-ai',
      author='Gianluca Lorenzo',
      author_email='gianluca.lorenzo@gmail.com',
      license='MIT',
      entry_points={'console_scripts':[
                        str(_program)+'=tileai.__main__:main']
      },
      keywords=[],
      tests_require=['pytest', 'coveralls'],
      zip_safe=False)
