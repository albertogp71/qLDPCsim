exec(open('yourpackage/version.py').read())
setup(
    name='yourpackage',
    version=__version__,
    # other setup parameters
)
