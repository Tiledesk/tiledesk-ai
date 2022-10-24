from setuptools import setup, find_packages
import glob
import os

with open('requirements.txt') as f:
    required = [x for x in f.read().splitlines() if not x.startswith("#")]

from tileai.cli import __version__, _program


setup(name=_program,
      version=__version__,
      packages=find_packages(include=['tileai','tileai.*']),
      description='tiledesk ai',
      url='https://github.com/Tiledesk/tiledesk-ai',
      author='Gianluca Lorenzo',
      author_email='gianluca.lorenzo@gmail.com',
      license='MIT',
      entry_points={'console_scripts':[
                        str(_program)+'=tileai.__main__:main']
      },
      keywords=[],
      install_requires=required,
      py_modules=['tileai.__main__'],
      tests_require=['pytest', 'coveralls'],
      zip_safe=False)
