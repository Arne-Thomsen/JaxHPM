from setuptools import find_packages, setup

setup(
    name="JaxHPM",
    version="0.0.1",
    url="https://github.com/Arne-Thomsen/JaxHPM",
    author="JaxHPM developers",
    description="Hybrid physical-neural extension to JaxPM for effective gas dynamics",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "jax_cosmo",
        "diffrax",
        "flax",
        "optax",
        "orbax",
        "h5py",
        "hdf5plugin",
        "tqdm",
        "matplotlib",
    ],
)
