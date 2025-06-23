from setuptools import find_packages, setup

setup(
    name='ML_project',
    version='0.0.1',
    author='Yashvardhan Singh',
    author_email='yashvardhansingh9532@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'seaborn',
        'matplotlib',
        'catboost',
        'xgboost',
        'scikit-learn',
        'dill',
        'flask'
    ]
)
