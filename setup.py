from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_descr_ = f.read()
    
with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

name_ = 'berryboxai'
decr_ = 'A command line tool for measuring quality traits from postharvest cranberries and blueberries.'

setup(
    name=name_,
    version='0.1.0.013',
    author='Jeff Neyhart, Collins Wakholi',
    author_email='jeffneyhart.work@gmail.com',
    description=decr_,
    long_description=long_descr_,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={
        'berryboxai': ['data/weights/*'],  # Include the models directory
    },
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'berryboxai=berryboxai.main:main',  # Entry point to the CLI tool
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
