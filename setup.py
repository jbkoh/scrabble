#from distutils.core import setup
from setuptools import setup, find_packages
from pkg_resources import parse_requirements
import pdb
__author__ = 'Jason Koh'
__version__ = '0.0.1'

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements(open('requirements.txt'))
#pdb.set_trace()
reqs = [ir.name for ir in install_reqs]

setup (
    name = 'scrabble',
    version = __version__,
    author = __author__,
    #packages = ['scrabble'],
    packages = find_packages(),
    description = 'Scrabble for building metadata normalization',
    include_package_data = True,
    scripts = ['scripts/scrabble'],
    data_files = [
        ('metadata', ['metadata/relationship_prior.json',
                      'metadata/bacnettype_mapping.csv',
                      'metadata/unit_mapping.csv',
                      ]),
        ('brick', ['brick/tags.json',
                   'brick/equip_tagsets.json',
                   'brick/location_tagsets.json',
                   'brick/point_tagsets.json',
                   'brick/location_subclass_dict.json',
                   'brick/point_subclass_dict.json',
                   'brick/equip_subclass_dict.json',
                   'brick/tagset_tree.json',
                   ])
    ],
    install_requires = reqs,
)
