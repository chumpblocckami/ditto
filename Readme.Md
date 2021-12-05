# Ditto

Ditto is a JAX implementation that leverages CLIP in order to generate Pokemon from an input text.

## Examples 
!image
!video

## Installation
> docker-compose build 
> docker-compose up
then go to localhost:5000 to be able to generate Pokemon.
If you want to run it without frontend, in ./backedn run ./run_naked.sh

## Hyperparameters:
- prompt
- scale
- eta
- batch_size
- image_size (this must be multiple of 8)
- steps
The default ones are a good tradeoff between speed and performance.

## Acknowledgments:
- (model)
- (code)

If you have any issue, feel free to dm or open a issue.

