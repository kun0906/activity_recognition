# pyenv vituralenv

# refer to 'https://realpython.com/intro-to-pyenv/'

$ cd activity_recognition $ pyenv versions

# create a virtual environment

# pyenv virtualenv <python_version> <environment_name>

$ pyenv virtualenv 3.7.9 ar

# Activating Your Versions

pyenv local ar pyenv which python pyenv activate ar

# deactivate your virtual environment

pyenv deactivate
