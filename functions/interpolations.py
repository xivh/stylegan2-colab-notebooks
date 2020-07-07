# modified from code written by https://github.com/arfafax
# needs to be cleaned up + condensed

"""
make_video - make an interpolation video grid
interpolate_between_seed - interpolate between two seeds and generate a video
def circular_interpolation - make a circular interpolation video
def generate_images - generate images from seeds
def blend_images - blend two seeds together
"""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import scipy
import math
from PIL import ImageFont
from PIL import ImageDraw
#from numpy import linalg
import moviepy.editor

# make an interpolation video grid
def make_video(grid_size = [4, 4], duration_sec = 60.0, mp4_fps = 20, random_seed=397, Gs=Gs):
    image_shrink = 1
    image_zoom = 1
    smoothing_sec = 1.0
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'
    mp4_file = 'random_grid_%s.mp4' % random_seed
    minibatch_size = 8

    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    # Generate latent vectors
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    import scipy
    all_latents = scipy.ndimage.gaussian_filter(all_latents,
                                                [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    def create_image_grid(images, grid_size=None):
        assert images.ndim == 3 or images.ndim == 4
        num, img_h, img_w, channels = images.shape

        if grid_size is not None:
            grid_w, grid_h = tuple(grid_size)
        else:
            grid_w = max(int(np.ceil(np.sqrt(num))), 1)
            grid_h = max((num - 1) // grid_w + 1, 1)

        grid = np.zeros([grid_h * img_h, grid_w * img_w, channels], dtype=images.dtype)
        for idx in range(num):
            x = (idx % grid_w) * img_w
            y = (idx // grid_w) * img_h
            grid[y : y + img_h, x : x + img_w] = images[idx]
        return grid

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.75,
                        randomize_noise=False, output_transform=fmt)

        grid = create_image_grid(images, grid_size)
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor
    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    video_clip.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)

    return mp4_file

# interpolate between two seeds and generate a video
def interpolate_between_seeds(seed_array, truncation, duration_sec = 10.0, smoothing_sec = 1.0, mp4_fps = 20, filename=None, text=False, Gs=Gs):
    #_G, _D, Gs = pickle.load(open("/content/network-e621.pkl", "rb"))
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    if seed_array[0] != seed_array[-1]:
        seed_array.append(seed_array[0])

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    synthesis_kwargs = dict(output_transform=Gs_kwargs.output_transform, truncation_psi=truncation, minibatch_size=8)
    if truncation is not None:
        Gs_kwargs.truncation_psi = truncation
    rnd = np.random.RandomState(seed_array[0])
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    batch_size = 1
    all_seeds = seed_array #[seed] * batch_size
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    #print(all_z)
    #print(all_z.shape)
    all_w = []

    labels = []
    for i, seed in enumerate(seed_array):
        z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:])])
        #print(i, seed, z)
        all_w_src = Gs.components.mapping.run(z, None) # [minibatch, layer, component]
        if truncation != 1:
            w_avg = Gs.get_var('dlatent_avg')
            all_w_src = w_avg + (all_w_src - w_avg) * truncation # [minibatch, layer, component]
        all_w.append(all_w_src)
    #print(all_w)
    #print(len(all_w))

    num_frames = int(np.rint(duration_sec * mp4_fps))
    def make_frame(t):
        blend = ((len(seed_array)-1)*t/duration_sec)%1.0
        src_i = math.floor((t/duration_sec)*(len(seed_array)-1))
        dst_i = src_i + 1
        #print(t, blend, src_i, dst_i)
        all_w_new = (blend * all_w[dst_i]) + (1 - blend) * all_w[src_i]
        all_images_src = Gs.components.synthesis.run(all_w_new, randomize_noise=False, **synthesis_kwargs)
        #all_images_dst = Gs.components.synthesis.run(all_w_dst, randomize_noise=False, **synthesis_kwargs)
        if text:
            new_im = PIL.Image.new('RGB', (512, 600))
            new_im.paste(PIL.Image.fromarray(np.median(all_images_src, axis=0).astype(np.uint8)), (0, 0))
            draw = ImageDraw.Draw(new_im)
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size=16)
            draw.text((10, 512), "{:0.2f}".format((1-blend)), (255, 0, 0), font=font)
            draw.text((50, 512), str(seed_array[src_i]), (255, 255, 255), font=font)
            draw.text((10, 550), "{:0.2f}".format((blend)), (0, 255, 0), font=font)
            draw.text((50, 550), str(seed_array[dst_i]), (255, 255, 255), font=font)
            return np.array(new_im)
        else:
            return all_images_src[0]

    import moviepy.editor
    mp4_file = 'interp_%s-%s.mp4' % (seed_array, truncation)
    if filename:
        mp4_file = filename
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'

    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    video_clip.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)

    return mp4_file

# make a circular interpolation video
def circular_interpolation(seed_a, seed_b, seed_c, radius = 40.0, Gs=Gs):
    rnd = np.random
    latents_a = np.random.RandomState(seed_a).randn(1, Gs.input_shape[1])
    latents_b = np.random.RandomState(seed_b).randn(1, Gs.input_shape[1])
    latents_c = np.random.RandomState(seed_c).randn(1, Gs.input_shape[1])

    def circ_generator(latents_interpolate):
        #radius = 40.0

        latents_axis_x = (latents_a - latents_b).flatten() / np.linalg.norm(latents_a - latents_b)
        latents_axis_y = (latents_a - latents_c).flatten() / np.linalg.norm(latents_a - latents_c)

        latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
        latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius

        latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
        return latents

    def mse(x, y):
        return (np.square(x - y)).mean()

    def generate_from_generator_adaptive(gen_func):
        max_step = 1.0
        current_pos = 0.0

        change_min = 10.0
        change_max = 11.0

        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

        current_latent = gen_func(current_pos)
        current_image = Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
        array_list = []

        video_length = 1.0
        while(current_pos < video_length):
            array_list.append(current_image)

            lower = current_pos
            upper = current_pos + max_step
            current_pos = (upper + lower) / 2.0

            current_latent = gen_func(current_pos)
            current_image = images = Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
            current_mse = mse(array_list[-1], current_image)

            while current_mse < change_min or current_mse > change_max:
                if current_mse < change_min:
                    lower = current_pos
                    current_pos = (upper + lower) / 2.0

                if current_mse > change_max:
                    upper = current_pos
                    current_pos = (upper + lower) / 2.0


                current_latent = gen_func(current_pos)
                current_image = images = Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
                current_mse = mse(array_list[-1], current_image)
            print(current_pos, current_mse)
        return array_list

    frames = generate_from_generator_adaptive(circ_generator)
    frames = moviepy.editor.ImageSequenceClip(frames, fps=30)

    # Generate video.
    print("Generating video")
    mp4_file = 'circular_%s-%s-%s-%s.mp4' % (seed_a, seed_b, seed_c, radius)
    #mp4_file = 'results/circular.mp4'
    mp4_codec = 'libx264'
    mp4_bitrate = '3M'
    mp4_fps = 20

    frames.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)
    return mp4_file

# generate images from seeds
def generate_images(seeds, truncation_psi, Gs=Gs):
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        #print(z)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save('seed%04d.png' % seed)
        display(PIL.Image.fromarray(images[0], 'RGB'))

# blend two seeds together
def blend_images(src_seed, dst_seed, blending=0.5, truncation_psi=0.7, Gs=Gs):
    #_G, _D, Gs = pickle.load(open("/content/network-e621.pkl", "rb"))
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    synthesis_kwargs = dict(output_transform=Gs_kwargs.output_transform, truncation_psi=truncation_psi, minibatch_size=8)
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    all_w = []

    for i, seed in enumerate([src_seed, dst_seed]):
        z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:])])
        #print(i, seed, z)
        all_w_src = Gs.components.mapping.run(z, None) # [minibatch, layer, component]
        if truncation_psi != 1:
            w_avg = Gs.get_var('dlatent_avg')
            all_w_src = w_avg + (all_w_src - w_avg) * truncation_psi # [minibatch, layer, component]
        all_w.append(all_w_src)

    w_new = (blending * all_w[0]) + (1 - blending) * all_w[1]
    images = Gs.components.synthesis.run(w_new, randomize_noise=False, **synthesis_kwargs)

    PIL.Image.fromarray(images[0], 'RGB').save('seed%04d.png' % seed)
    display(PIL.Image.fromarray(images[0], 'RGB'))
