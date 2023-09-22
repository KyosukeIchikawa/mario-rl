from setuptools import setup, find_packages


setup(
    name="mario_rl",
    author='KyosukeIchikawa',
    url='https://github.com/KyosukeIchikawa/mario-rl',
    packages=find_packages(exclude=['test*']),
    include_package_data=True,
)
