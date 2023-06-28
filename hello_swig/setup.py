#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2020-09-26 18:30
from distutils.core import setup, Extension

add_module = Extension("_add", sources=["add_wrap.cxx", "add.cc"],)

setup(
    name="add",  # 打包后的名称
    version="0.1",
    author="SWIG Docs",
    description="Simple swig pht from docs",
    ext_modules=[add_module],  # 与上面的扩展模块名称一致
    py_modules=["add"],  # 需要打包的模块列表
)
