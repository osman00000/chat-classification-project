from setuptools import setup

setup(
    name='chatclassifier',
    version='0.0.1',
    py_modules=['chatclassifier.classifier'],
    install_requires=[
        'transformers',
        'importlib-metadata; python_version>="3.10"',
    ],
)

