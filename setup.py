from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as f:
    readme = f.read()


setup(
      name="Torch-DCA",
      version="0.0.1",
      author="Oscar Davalos, A. Ali Heydari",
      author_email="odavalos2@ucmerced.edu",
      description="A PyTorch implementation of DCA: Deep count autoencoder for denoising scRNA-seq data",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/odavalos/Torch-DCA",
      download_url="https://github.com/odavalos/Torch-DCA",
      packages=find_packages(),
      keywords=['Single-cell RNA-seq', 'scRNA-seq','Clustering', 'Neural Networks', 'Autoencoders', 'Tabular Data'],
      install_requires=[
                        'scanpy>=1.8.1',
                        'numpy>=1.20.3',
                        'torch>=1.10.2'
                        ]
)
