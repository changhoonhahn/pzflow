from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).parent
# Keep installation dependencies in the single shared requirements file.
requirements = [
    line.strip()
    for line in (root / 'requirements.txt').read_text().splitlines()
    if line.strip() and not line.lstrip().startswith('#')
]
# Discover only the small package under src/; analysis entry points stay in bin/.
setup(
    name='pzflow-cosmos',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.12',
    install_requires=requirements,
    description='Probabilistic COSMOS2020 redshift transfer to broadband photometry',
    long_description=(root / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    license='MIT',
)
