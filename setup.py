from setuptools import setup, find_packages
from setuptools.command.install import install as _install
import os
import subprocess
import platform

with open('README.md', encoding='utf-8') as f:
    long_descr_ = f.read()
    
with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

name_ = 'berryboxai'
decr_ = 'A command line tool for measuring quality traits from postharvest cranberries and blueberries.'


class InstallCommand(_install):
    """Custom installation command to convert model weights."""
    
    def run(self):
        # Run the standard installation process
        _install.run(self)
        
        # Check if the system is a PC (Windows) and convert weights
        if platform.system() == 'Windows':
            self.convert_model()

    def convert_model(self):
        # Path to your weights file
        input_weights_path1 = os.path.join(os.path.dirname(__file__), 'berryboxai/data/weights/berrybox_berry-seg.pt')
        input_weights_path2 = os.path.join(os.path.dirname(__file__), 'berryboxai/data/weights/berrybox_rot-det.pt')

        # Convert weights to OpenVINO format using Ultralytics
        subprocess.run([
            'python', 'convert_model.py', 
            '--input', input_weights_path1, 
        ], check=True)

        # Convert weights to OpenVINO format using Ultralytics
        subprocess.run([
            'python', 'convert_model.py', 
            '--input', input_weights_path2, 
        ], check=True)
        
        print('Model weights converted to OpenVINO format.')

setup(
    name=name_,
    version='0.1.0',
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
    cmdclass={
        'install': InstallCommand,
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
