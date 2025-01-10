from setuptools import setup, find_packages

setup(
    name="crystfelparser",
    version="0.0.1",
    description="crystfelparser",
    classifiers=["Development Status :: 2 - Pre-Alpha",
                 "Topic :: Scientific/Engineering :: File processing"],
    keywords=[],
    url="https://github.com/pgasparo/crystfelparser",
    author="Piero Gasparotto",
    author_email="piero.gasparotto@gmail.com",
    license="AGPLv3",
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'crystfelparser = crystfelparser.crystfelparser:main',
            'xdsparser = crystfelparser.xdsparser:main',
        ],
    },
)
