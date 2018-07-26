from collections import namedtuple

Pokemon = namedtuple('Pokemon', 'name label')

squirtle = Pokemon('squirtle', 0)
bulbasaur = Pokemon('bulbasaur', 1)
charmander = Pokemon('charmander', 2)
pikachu = Pokemon('pikachu', 3)

pokemons = [squirtle, bulbasaur, charmander, pikachu]
DATASET_PATH = 'dataset'
