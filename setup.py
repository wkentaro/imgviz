from __future__ import print_function

import distutils.spawn
import os
import os.path as osp
import re
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


def get_version():
    filename = "imgviz/__init__.py"
    with open(filename) as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def get_install_requires():
    install_requires = []
    with open("requirements.txt") as f:
        for req in f:
            install_requires.append(req.strip())
    return install_requires


def get_extras_require():
    extras_require = {
        "all": [
            "scikit-image",
            "scikit-learn",
        ]
    }
    if sys.version_info.major == 2:
        extras_require["all"].append("opencv-python<3.4.10.37")
        extras_require["all"].append("pyglet<1.5")
    elif sys.version_info.major == 3 and sys.version_info.minor < 8:
        extras_require["all"].append("opencv-python")
        extras_require["all"].append("pyglet<2")
    else:
        extras_require["all"].append("opencv-python")
        extras_require["all"].append("pyglet")
    return extras_require


def get_long_description():
    with open("README.md") as f:
        long_description = f.read()
    try:
        # when this package is being released
        import github2pypi

        return github2pypi.replace_url(
            slug="wkentaro/imgviz", content=long_description, branch="main"
        )
    except ImportError:
        # when this package is being installed
        return long_description


def get_package_data():
    package_data = []
    for dirpath, dirnames, filenames in os.walk("data"):
        for filename in filenames:
            data_file = osp.join(dirpath, filename)
            data_file = osp.join(osp.split(data_file)[1:])
            package_data.append(data_file)
    return {"imgviz": package_data}


def main():
    version = get_version()

    if sys.argv[1] == "release":
        try:
            import github2pypi  # NOQA
        except ImportError:
            print(
                "Please install github2pypi\n\n\tpip install github2pypi\n",
                file=sys.stderr,
            )
            sys.exit(1)

        if not distutils.spawn.find_executable("twine"):
            print(
                "Please install twine:\n\n\tpip install twine\n",
                file=sys.stderr,
            )
            sys.exit(1)

        commands = [
            "git pull origin main",
            "git tag v{:s}".format(version),
            "git push origin main --tags",
            "python setup.py sdist",
            "twine upload dist/imgviz-{:s}.tar.gz".format(version),
        ]
        for cmd in commands:
            print("+ {}".format(cmd))
            subprocess.check_call(shlex.split(cmd))
        sys.exit(0)

    setup(
        name="imgviz",
        version=version,
        packages=find_packages(),
        install_requires=get_install_requires(),
        extras_require=get_extras_require(),
        description="Image Visualization Tools",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        package_data=get_package_data(),
        include_package_data=True,
        author="Kentaro Wada",
        author_email="www.kentaro.wada@gmail.com",
        url="http://github.com/wkentaro/imgviz",
        license="MIT",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
    )


if __name__ == "__main__":
    main()
