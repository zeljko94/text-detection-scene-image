import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opencv_text_detection",
    version="0.0.1",
    author="Tom Hoag & Adrian Rosebrock",
    description="OpenCV efficient and accurate scene text detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/tomhoag/opencv-text-detection",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'nms>=0.1.5',
        'numpy==1.15.0',
        'opencv-python>=3.4.3.18',
    ],
    entry_points={'console_scripts': [
        'opencv-text-detection = opencv_text_detection.text_detection:text_detection_command',
    ]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)