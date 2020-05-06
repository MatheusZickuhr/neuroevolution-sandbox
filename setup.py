from distutils.core import setup

setup(
    name='neuroevolution-sandbox',
    packages=['neuroevolution_sandbox', 'neuroevolution_sandbox/agents',  'neuroevolution_sandbox/env_adapters'],
    version='0.1',
    license='MIT',
    description='A neuroevolution sandbox with NE and NEAT agents. Compatible with gym and ple with the option to add more envs.',
    author='Matheus Zickuhr',
    author_email='matheuszickuhr97@gmail.com',
    url='https://github.com/MatheusZickuhr/neuroevolution-sandbox',
    keywords=['neuroevolution', 'neat', 'ne', 'ann', 'deep-learning'],
    install_requires=['python-ne', 'neat-python'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
