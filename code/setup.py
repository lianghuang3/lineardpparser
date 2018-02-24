from distutils.core import setup
from distutils.extension import Extension

setup(name = "_mycollections",
      version = "1.0",
      ext_modules = [Extension("_mycollections", ["_mycollectionsmodule.c"])]
      )

setup(name = "mydouble",
      version = "1.0",
      ext_modules = [Extension("mydouble", ["mydouble.c"])]
      )
