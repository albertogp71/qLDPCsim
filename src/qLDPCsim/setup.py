exec(open('qLDPCsim/version.py').read())
setup(
    name='qLDPCsim',
    version=__version__,
    description='qLDPCsim is a toolkit for performance evaluation of quantum LDPC (CSS-type) error correction codes.',
    author='AlbertoGP71',
    author_email='alberto.g.perotti@gmail.com',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license='LICENSE',
)
