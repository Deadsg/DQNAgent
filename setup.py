from setuptools import setup, find_packages

setup(
    name="dqnagent", # The name of your project
    version="0.0.1", # The version of your project
    description="A Python project that uses deep Q-networks and self-learning Q loops to create adaptive and robust agents for different environments.", # A brief description of your project
    long_description=open("README.md").read(), # A long description of your project from the README.md file
    long_description_content_type="text/markdown", # The content type of your long description
    url="https://github.com/Deadsg/DQNAgent", # The URL of your project's homepage
    author="DQNAgent", # The name of the author of your project
    author_email="hello@dqn.ai", # The email of the author of your project
    license="MIT", # The license of your project
    classifiers=[ # A list of classifiers that describe your project
        "Development Status :: 3 - Alpha", # The development status of your project
        "Intended Audience :: Developers", # The intended audience of your project
        "Topic :: Software Development :: Libraries", # The topic of your project
        "License :: OSI Approved :: MIT License", # The license of your project
        "Programming Language :: Python :: 3", # The programming language of your project
        "Programming Language :: Python :: 3.8", # The specific Python version of your project
        "Programming Language :: Python :: 3.9", # The specific Python version of your project
    ],
    packages=find_packages(where="src"), # A list of packages that are included in your project
    package_dir={"": "src"}, # A mapping of package names to directories
    python_requires=">=3.8, <4", # The Python version requirement of your project
    install_requires=[ # A list of dependencies that are required for your project
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "tensorflow",
        "scikit-learn"
    ],
    extras_require={ # A mapping of optional dependencies that are grouped by features
        "dev": [ # Dependencies for development
            "pytest",
            "flake8",
            "black",
            "isort",
            "mypy",
            "sphinx",
        ],
         "docs": [ # Dependencies for documentation
            "sphinx",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints",
        ],
    },
    entry_points={ # A mapping of entry points that define how to run your project
        "console_scripts": [ # A list of console scripts that can be invoked from the command line
            "qllm = DQN_Node_Agent.__main__:main", # The name and the function of the script
        ],
    },
    include_package_data=True, # A flag that indicates whether to include non-code files in your package
    zip_safe=False, # A flag that indicates whether your project can be safely installed in a zip file
)