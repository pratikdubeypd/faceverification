from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='faceverification',
    version='1.0',
    description='A Face Verification module using keras VGGFace',
    long_description='A face verification module which compares the faces present in 2 images. This is the initial release.',
    url='',  
    author='Pratik Dubey',
    author_email='pratikpddubey@gmail.com',
    license='MIT', 
    classifiers=classifiers,
    keywords=['face verification', 'keras', 'vggface', 'deeplearning', 'face recognition'], 
    packages=find_packages(),
    install_requires=[
        'keras', 'numpy>=1.9.1', 'scipy>=0.14', 'h5py', 
        'pillow', 'mtcnn', 'keras-vggface'
    ],
    extras_require={
        "tf": ["tensorflow"],
        "tf_gpu": ["tensorflow-gpu"],
    }
)