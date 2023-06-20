from setuptools import setup, find_packages

setup(
    name='hamzaGlyph',
    version='1.1.0',
    description='hieroglyph translation library',
    author='Mahmoud Hamza Mohamed Ibrahim Abdella',
    license='MIT',
    email='mahmoud.hamza1592001@gmail.com',
    packages=find_packages(),
    install_requires=[
        'yolov5',
        'cv2',
        'difflib'
    ]
)
