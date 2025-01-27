from setuptools import setup, find_packages

setup(
    name="janus_prot",  # 你的项目名称
    version="0.1",  # 版本号
    packages=find_packages(),  # 自动查找所有包
    install_requires=[  # 依赖项（如果有）
        "torch",
        "transformers",
        "deepspeed",
        "tokenizers",
        "datasets",
        "tqdm",
    ],
)