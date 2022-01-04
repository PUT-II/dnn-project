from collections import namedtuple

make_episode = namedtuple(
    typename='Episode',
    field_names=[
        'states',
        'actions',
        'infos',
        'rewards',
        'init_command',
        'total_return',
        'length',
    ]
)
