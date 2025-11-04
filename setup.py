import os

from sysconfig import get_path
from distutils.core import Extension, setup

try:
    from Cython.Build import cythonize
except Exception:
    print("Por favor instale cython e tente novamente.")
    raise SystemExit

PACKAGES = [
    'biblioteca',
    'biblioteca.estruturas',
    'biblioteca.helpers'
]

# python setup.py build_ext --inplace
def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(root, 'biblioteca/__about__.py')) as f:
        about = {}
        exec(f.read(), about)

    # with open(os.path.join(root, 'README.rst')) as f:
    #     readme = f.read()

    extensions = []
    extensions.append(
        Extension(
            "biblioteca.helpers.hash.city",
            language='c++',
            sources=[
                os.path.join('biblioteca', 'helpers', 'hash', 'city.pyx'),
                os.path.join('biblioteca', 'helpers', 'hash', 'src', 'city.cpp')
            ],
            include_dirs=[
                get_path("include"),
                os.path.join('biblioteca', 'helpers', 'hash', 'src')
            ]
        )
    )
    extensions.append(
        Extension(
            "biblioteca.helpers.storage.bitvector",
            language='c++',
            sources=[
                os.path.join('biblioteca', 'helpers', 'storage', 'bitvector.pyx'),
                os.path.join('biblioteca', 'helpers', 'storage', 'src', 'BitField.cpp')
            ],
            include_dirs=[
                get_path("include"),
                os.path.join('biblioteca', 'helpers', 'storage', 'src')
            ]
        )
    )
    extensions.append(
        Extension(
            "biblioteca.estruturas.cms_cu",
            language='c',
            sources=[os.path.join('biblioteca', 'estruturas', 'cms_cu.pyx')],
            include_dirs=[
                get_path("include"),
            ]
        )
    )
    extensions.append(
        Extension(
            "biblioteca.estruturas.cmls_cu",
            language='c',
            sources=[os.path.join('biblioteca', 'estruturas', 'cmls_cu.pyx')],
            include_dirs=[
                get_path("include"),
            ]
        )
    )
    extensions.append(
        Extension(
            "biblioteca.estruturas.cmls_cu_h",
            language='c',
            sources=[os.path.join('biblioteca', 'estruturas', 'cmls_cu_h.pyx')],
            include_dirs=[
                get_path("include"),
            ]
        )
    )
    extensions.append(
        Extension(
            "biblioteca.estruturas.cl",
            language='c',
            sources=[os.path.join('biblioteca', 'estruturas', 'cl.pyx')],
            include_dirs=[
                get_path("include"),
            ]
        )
    )
    extensions.append(
        Extension(
            "biblioteca.estruturas.cmts",
            language='c++',
            sources=[os.path.join('biblioteca', 'estruturas', 'cmts.pyx')],
            include_dirs=[
                get_path("include"),
            ]
        )
    )

    setup(
        name="biblioteca",
        packages=PACKAGES,
        package_data={'': ['*.pyx', '*.pxd', '*.cpp', '*.h']},
        description=about['__summary__'],
        long_description='Biblioteca para TCC',
        keywords=about['__keywords__'],
        author=about['__author__'],
        author_email=about['__email__'],
        version=about['__version__'],
        url=about['__uri__'],
        license=about['__license__'],
        ext_modules=cythonize(
            extensions,
            compiler_directives={"language_level": "3str"}
        ),
        python_requires='>=3.12.7',
        install_requires=["cython>=3.1.5"]
    )


if __name__ == '__main__':
    setup_package()