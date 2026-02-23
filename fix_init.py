import os

dirs = [
    'app', 'app/core', 'app/mqtt', 'app/projects',
    'app/projects/assets', 'app/projects/vehicles',
    'app/projects/epi_check', 'app/projects/epi_check/api',
    'app/projects/epi_check/models', 'app/projects/epi_check/engine',
    'app/streaming', 'app/ui'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, '__init__.py')
    if not os.path.exists(f):
        open(f, 'w').close()
        print('Criado:', f)
    else:
        print('OK:', f)