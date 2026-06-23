from setuptools import find_packages, setup


setup(
    name="adaptive-transient-detector",
    version="0.1.0",
    description="A small command-line tool for spotting audio transients with adaptive multi-representation analysis.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy>=1.20"],
    entry_points={
        "console_scripts": [
            "atd=adaptive_transient_detector.cli:main",
        ]
    },
)
