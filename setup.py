from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
   name='enlp',
   version='0.1',
   description='efficient-NLP',
   author='David Rau, Nikos Kondylidis',
   author_email='{d.m.rau, n.kondylidis}@uva.nl',
   install_requires=requirements,
   packages=['enlp'],  #same as name
)
