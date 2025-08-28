import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    description = fh.read()

setuptools.setup(
    name = 'pynatple',
    version = '0.0.1',
    author = 'Alfa',
    author_email = 'm.alfaxx08@gmail.com',
    packages = ['pynatple'],
    description = 'A Python library for gravity inversion and optimization',
    long_description = description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/MayaYvano/pynatple',
    license = 'MIT',
    python_requires = '>=3.7',
    install_requires = [
        'numpy',
        'scipy',
        'xarray',
        'xrft',
        'pandas',
    ],
)