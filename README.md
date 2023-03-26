# ChaosVAE
ChaosVAE is a deep learning model based on Variational Autoencoders (VAEs) that can learn and generate chaotic systems. It is designed to learn the dynamics of complex systems from time series data and can be used to generate new samples from the learned dynamics.

# installation
## Clone the repository and navigate to the project directory:

git clone https://github.com/username/ChaosVAE.git

cd ChaosVAE
## Create a new virtual environment and activate it:

python3 -m venv env
source env/bin/activate

## dataset:
you can generate it using:

the subroutine rkdumb() taken from Numerical Recipes, with a step size of 0.01.

from the lorenz equations:

dx/dt = sigma * (y - x)

dy/dt = r * x - y - x * z

dz/dt = x * y - b * z

Or you can find the data here:

https://physics.emory.edu/faculty/weeks/research/tseries1.html

## License
This project is licensed under the MIT License.

## This project is Incompleted:

I'm still working on the project and the rest of it is local and private.


If you want to help send me a message!
