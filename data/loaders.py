import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_digits, load_iris, load_wine, fetch_covtype
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision import datasets, transforms
from ucimlrepo import fetch_ucirepo


def load_covertype():
    """
    Загрузка датасета Covertype из sklearn или другого источника.
    """
    # Пример загрузки из OpenML
    covertype = fetch_covtype()
    X, y = covertype.data, covertype.target.astype(int)
    return X, y

def load_pendigits():
    """
    Загрузка датасета PenDigits.
    """
    pendigits = fetch_openml(name='pendigits', version=1, as_frame=False)
    X, y = pendigits.data, pendigits.target.astype(int)
    return X, y

def load_letters():
    """
    Загрузка датасета Letters.
    """
    letters = fetch_openml(name='letter', version=1, as_frame=False)
    X, y = letters.data, letters.target
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded
def load_mnist():
    """
    Загрузка датасета MNIST.
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    return X, y

def load_usps():
    """
    Загрузка датасета USPS.
    """
    # USPS не доступен напрямую через sklearn, можно использовать torchvision
    transform = transforms.Compose([transforms.ToTensor()])
    usps = datasets.USPS(root='./data/raw', train=True, download=True, transform=transform)
    X = usps.data.reshape(usps.data.shape[0], -1)
    y = np.array(usps.targets)
    return X, y

def load_fashion_mnist():
    """
    Загрузка датасета FashionMNIST.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    fashion_mnist = datasets.FashionMNIST(root='./data/raw', train=True, download=True, transform=transform)
    X = fashion_mnist.data.numpy().reshape(fashion_mnist.data.shape[0], -1)
    y = fashion_mnist.targets.numpy()
    return X, y

def load_cifar10():
    """
    Загрузка датасета CIFAR-10.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10 = datasets.CIFAR10(root='./data/raw', train=True, download=True, transform=transform)
    X = cifar10.data.reshape(cifar10.data.shape[0], -1)
    y = np.array(cifar10.targets)
    return X, y

# def load_kannada_mnist():
#     """
#     Загрузка датасета Kannada MNIST.
#     """
#     # Если датасет доступен через OpenML или другой источник
#     kannada = fetch_openml(name='kannada-mnist', version=1, as_frame=False)
#     X, y = kannada.data, kannada.target.astype(int)
#     return X, y

def get_dataset(name):
    """
    Функция для получения выбранного датасета.
    """
    loaders = {
        'Covertype': load_covertype,
        'PenDigits': load_pendigits,
        'Letters': load_letters,
        'MNIST': load_mnist,
        'USPS': load_usps,
        'FashionMNIST': load_fashion_mnist,
        'CIFAR10': load_cifar10,
        # 'KannadaMNIST': load_kannada_mnist
    }
    if name not in loaders:
        raise ValueError(f"Dataset {name} is not supported.")
    X, y = loaders[name]()
    # Предварительная обработка
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

