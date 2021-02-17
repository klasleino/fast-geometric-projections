from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open('requirements.txt', 'r') as requirements:
    setup(
        name='fgp-cert',
        version='0.0.1',
        install_requires=list(requirements.read().splitlines()),
        packages=find_packages(),
        description=
            'tool for certifying local robustness in deep networks',
        python_requires='>=3.6',
        author='Klas Leino',
        author_email='kleino@cs.cmu.edu',
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License'],
        long_description=long_description,
        long_description_content_type='text/markdown'
    )
