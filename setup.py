import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyAudioMixer", # Replace with your own username
    version="0.0.1",
    author="Jean-Pierre Coetzee",
    author_email="jeanpierrec19@gmail.com",
    description="Advanced Realtime Software Mixer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jpc0/PyAudioMixer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)