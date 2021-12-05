import math 
import jax
import jaxtorch
import jax.numpy as jnp
from jaxtorch import Module, PRNG, Context, ParamState, nn, init
import clip_jax
import numpy as np
import utils 
from functools import partial 

## Define the model (a residual U-Net)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return self.main(cx, input) + self.skip(cx, input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
            nn.ReLU(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return jnp.concatenate([self.main(cx, input), self.skip(cx, input)], axis=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = init.normal(out_features // 2, in_features, stddev=std)

    def forward(self, cx, input):
        f = 2 * math.pi * input @ cx[self.weight].T
        return jnp.concatenate([f.cos(), f.sin()], axis=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].broadcast_to(input.shape[:2] + shape[2:])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(10, 4)

        self.net = nn.Sequential(
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.image.Downsample2d(),  # 64x64 -> 32x32
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.image.Downsample2d(),  # 32x32 -> 16x16
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    SkipBlock([
                        nn.image.Downsample2d(),  # 16x16 -> 8x8
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.image.Upsample2d(),
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    nn.image.Upsample2d(),
                ]),
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.image.Upsample2d(),            # Haven't implemented ConvTranpose2d yet.
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout=False),
        )

    def forward(self, cx, input, log_snrs, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(cx, log_snrs[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cx, cond), input.shape)
        return self.net(cx, jnp.concatenate([input, class_embed, timestep_embed], axis=1))


# Define the noise schedule

def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -jnp.expm1(1e-4 + 10 * t**2).log()


def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    alphas_squared = jax.nn.sigmoid(log_snrs)
    sigmas_squared = jax.nn.sigmoid(-log_snrs)
    return alphas_squared.sqrt(), sigmas_squared.sqrt()

class Gan():
    def __init__(self,):
        print('Using device:', jax.devices())

        self.model = Diffusion()
        self.params_ema = self.model.init_weights(jax.random.PRNGKey(0))
        print('Model parameters:', sum(np.prod(p.shape) for p in self.params_ema.values.values()))

        self.image_fn, self.text_fn, self.clip_params, self.clip_size, self.normalize = self.load_clip

# Load checkpoint
    def load_state(self):
        #class StateDict(dict):
        #    pass
        state_dict = jaxtorch.pt.load(utils.fetch_model('https://set.zlkj.in/models/diffusion/pokemon_diffusion_gen3+4_c64_6783.pth'))
        self.model.load_state_dict(self.model, state_dict['model_ema'], strict=False)

    def load_clip(self):
        print('Loading CLIP model...')
        image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/32')
        clip_size = 224
        normalize = utils.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711])
        return image_fn, text_fn, clip_params, clip_size, normalize

    ## Define model wrappers

    @jax.jit
    def eval_model(self, params, xs, ts, classes, key):
        cx = Context(params, key).eval_mode_()
        return self.model(cx, xs, ts, classes)

    def txt(self, prompt):
        """Returns normalized embedding."""
        text = clip_jax.tokenize([prompt])
        text_embed = self.text_fn(self.clip_params, text)
        return utils.norm1(text_embed.reshape(512))

    def emb_image(self, image, clip_params=None):
        return utils.norm1(self.image_fn(clip_params, image))

    def base_cond_fn(self, x, t, text_embed, clip_guidance_scale, classes, key, params_ema, clip_params):
        rng = PRNG(key)
        n = x.shape[0]

        log_snrs = get_ddpm_schedule(t)
        alphas, sigmas = get_alphas_sigmas(log_snrs)

        def denoise(x, key):
            eps = utils.eval_model(params_ema, x, log_snrs.broadcast_to([n]), classes, rng.split())
            # Predict the denoised image
            pred = (x - eps * sigmas) / alphas
            x_in = pred * sigmas + x * alphas
            return x_in
        
        x_in, backward = jax.vjp(partial(denoise, key=rng.split()), x)

        def clip_loss(x_in):
            x_in = jax.image.resize(x_in, [n, 3, 224, 224], method='nearest')
            clip_in = self.normalize(x_in.add(1).div(2))
            image_embeds = self.emb_image(clip_in, clip_params).reshape([n, 512])
            losses = utils.spherical_dist_loss(image_embeds, text_embed)
            return losses.sum() * clip_guidance_scale
        clip_grad = jax.grad(clip_loss)(x_in)

        return -backward(clip_grad)[0]
        
    base_cond_fn = jax.jit(base_cond_fn)


    def run(self, seed, prompt, clip_guidance_scale, eta, batch_size, image_size, steps):
        ## Actually do the run

        def cond_fn(*args, **kwargs):
            grad = self.base_cond_fn(*args, **kwargs)
            # Gradient nondeterminism monitoring
            # grad2 = base_cond_fn(*args, **kwargs)
            # average = (grad + grad2) / 2
            # print((grad - grad2).abs().mean() / average.abs().mean())
            return grad

        rng = PRNG(jax.random.PRNGKey(seed))

        fakes = jax.random.normal(rng.split(), [batch_size, 3, image_size, image_size])

        fakes_classes = jnp.array([0] * batch_size) # plain
        ts = jnp.ones([batch_size])

        # Create the noise schedule
        t = jnp.linspace(1, 0, steps + 1)[:-1]
        log_snrs = get_ddpm_schedule(t)
        alphas, sigmas = get_alphas_sigmas(log_snrs)

        # The sampling loop
        for i in trange(steps):

            # Get the model output (eps, the predicted noise)
            eps = self.eval_model(self.params_ema, fakes, ts * log_snrs[i], fakes_classes, rng.split())

            # Predict the denoised image
            pred = (fakes - eps * sigmas[i]) / alphas[i]

            # If we are not on the last timestep, compute the noisy image for the
            # next timestep.
            if i < steps - 1:

                cond_score = cond_fn(fakes, t[i], self.text_embed, clip_guidance_scale, fakes_classes, rng.split(), params_ema, clip_params)

                eps = eps - sigmas[i] * cond_score
                pred = (fakes - eps * sigmas[i]) / alphas[i]

                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
                adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

                # Recombine the predicted noise and predicted denoised image in the
                # correct proportions for the next step
                fakes = pred * alphas[i + 1] + eps * adjusted_sigma

                # Add the correct amount of fresh noise
                if eta:
                    fakes += jax.random.normal(rng.split(), fakes.shape) * ddim_sigma

            # If we are on the last timestep, output the denoised image
            else:
                fakes = pred
        ### qua da capire come e cosa mostrare una volta settato il tutto bene
        grid = utils.make_grid(torch.tensor(np.array(fakes)), 4).cpu()
        timestring = time.strftime('%Y%m%d%H%M%S')
        os.makedirs('samples', exist_ok=True)
        filename = f'samples/{timestring}_{prompt}.png'
        TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
        display.display(display.Image(filename))
        print(f'Saved {filename}')