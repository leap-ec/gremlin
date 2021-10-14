from setuptools import setup
# Load the version number from inside the package
exec(open('gremlin/__version__.py').read())

# Use the README as the long_description
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='gremlin',
    version=__version__,
    packages=['gremlin'],
    entry_points={
        'console_scripts': [
            'gremlin = gremlin.gremlin:client'
        ],
    },
    url='https://github.com/piprrr/gremlin',
    license='MIT License',
    author='Mark Coletti',
    author_email='colettima@ornl.gov',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description=('Adversarial evolutionary algorithm for'
                 'training data optimization')
)
