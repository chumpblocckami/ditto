import requests
import io 
import os 
import jax.numpy as jnp

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def fetch_model(url_or_path):
    basename = os.path.basename(url_or_path)
    if os.path.exists(basename):
        return basename
    else:
        data = fetch(url_or_path).read()
        with open(basename, 'wb') as fp:
            fp.write(data)
        return basename

def Normalize(mean, std):
    mean = jnp.array(mean).reshape(3,1,1)
    std = jnp.array(std).reshape(3,1,1)
    def forward(image):
        return (image - mean) / std
    return forward

def norm1(x):
    """Normalize to the unit sphere."""
    return x / x.square().sum(axis=-1, keepdims=True).sqrt()

def spherical_dist_loss(x, y):
    x = norm1(x)
    y = norm1(y)
    return (x - y).square().sum(axis=-1).sqrt().div(2).arcsin().square().mul(2)