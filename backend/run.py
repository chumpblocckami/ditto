## Settings for the run
seed = 0

# Prompt for CLIP guidance
prompt = 'a pokemon resembling a squid #pixelart'
text_embed = txt(prompt)

# Strength of conditioning
clip_guidance_scale = 2000

# The amount of noise to add each timestep when sampling
# 0 = no noise (DDIM)
# 1 = full noise (DDPM)
eta = 1.0

batch_size = 16

# Image size. Was trained on 64x64. Must be a multiple of 8 but different sizes are possible.
image_size = 64

# Number of steps for sampling, more = better quality generally
steps = 250

## Actually do the run

def cond_fn(*args, **kwargs):
  grad = base_cond_fn(*args, **kwargs)
  # Gradient nondeterminism monitoring
  # grad2 = base_cond_fn(*args, **kwargs)
  # average = (grad + grad2) / 2
  # print((grad - grad2).abs().mean() / average.abs().mean())
  return grad

def demo():
    tqdm.write('Sampling...')
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
        eps = eval_model(params_ema, fakes, ts * log_snrs[i], fakes_classes, rng.split())

        # Predict the denoised image
        pred = (fakes - eps * sigmas[i]) / alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:

            cond_score = cond_fn(fakes, t[i], text_embed, clip_guidance_scale, fakes_classes, rng.split(), params_ema, clip_params)

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

    grid = utils.make_grid(torch.tensor(np.array(fakes)), 4).cpu()
    timestring = time.strftime('%Y%m%d%H%M%S')
    os.makedirs('samples', exist_ok=True)
    filename = f'samples/{timestring}_{prompt}.png'
    TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
    display.display(display.Image(filename))
    print(f'Saved {filename}')

demo()