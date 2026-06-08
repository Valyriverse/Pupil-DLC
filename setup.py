from setuptools import setup, find_packages

setup(
    name='pupil_dlc',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'deeplabcut',
        'pandas',
        'numpy',
        'scikit-image',
        'click',
        'pyfiglet',
        'pyyaml',
        'py7zr',
        'requests',
    ],
    entry_points={
        'console_scripts': [
            'pupil-dlc = pupil_dlc.pupil_dlc:main'
        ]
    }
)

