import cv2 as cv
import numpy as np
import os
from constants import pokemons, DATASET_PATH

"""
input [mean blue, mean green, mean red, main color, hu moments]
label pokemon label
"""


train_data = []
labels = []


def _get_total_images(pokemon) -> int:
    return len(os.listdir(f'{DATASET_PATH}/images/{pokemon.name}'))


def _generate_data(pokemon: 'a pokemon'):
    total_img = _get_total_images(pokemon)

    for i in range(total_img):
        image = _open_img(pokemon, i)
        blue, green, red = _get_colors(image)
        mean_blue, mean_green, mean_red = _get_mean_color(blue, green, red)
        main_color = _get_main_color(mean_blue, mean_green, mean_red)
        moments = _get_hu_moments(image)
        train_data.append([mean_blue, mean_green, mean_red, main_color, *moments])
        labels.append(pokemon.label)


def load_training_data():
    for pokemon in pokemons:
        _generate_data(pokemon)
    return train_data, labels


def _open_img(pokemon: 'a pokemon', index: int, flag=1) -> np.ndarray:
    path = f'{DATASET_PATH}/images/{pokemon.name}/{pokemon.name}-{index}.png'
    return cv.imread(path, flags=flag)


def _get_colors(image: 'image representation') -> tuple:
    blue, green, red = cv.split(image)
    return blue.ravel(), green.ravel(), red.ravel()


def _get_mean_color(blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> 'tuple of mean colors':
    mean_blue = np.mean(blue)
    mean_green = np.mean(green)
    mean_red = np.mean(red)
    return mean_blue, mean_green, mean_red


def _get_main_color(mean_blue: np.ndarray, mean_green: np.ndarray, mean_red: np.ndarray) -> np.float64:
    if mean_blue > mean_green and mean_blue > mean_green: return -1
    if mean_green > mean_blue and mean_green > mean_red: return 0
    return 1


def _get_hu_moments(image: 'a image representation'):
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    moments = cv.moments(gray_img)
    hu_moments = cv.HuMoments(moments)
    return hu_moments.flatten()
