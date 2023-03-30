import hashlib
import math
import random
import secrets

import cv2
import numpy as np
from perlin_noise import PerlinNoise
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def _perlin_blob(seed=42):
    size = 128
    # This frequency seems to work fine for blobs
    freq = 333

    # We draw two circular gradient on top of the perlin noise
    # to make sure there is a blob in the center of the image
    max_gradient_dist_1 = 64
    max_gradient_dist_2 = 32
    gradient_weight_1 = 0.333
    gradient_weight_2 = 0.1666

    # We make the noise a bit more bright to produce larger blobs
    perlin_noise_boost = 0.1666

    # Create a new Perlin noise object
    noise = PerlinNoise(octaves=4, seed=seed)

    # Create a new PIL image with the given size and mode
    blob_image = Image.new("L", (size, size))

    # Loop over each pixel in the image and set its value based on the Perlin noise
    for x in range(size):
        for y in range(size):
            value = noise([x / freq, y / freq])
            # The noise is between -0.5 and 0.5, so we add 0.5 to make it between 0 and 1
            value = value + 0.5 + perlin_noise_boost

            # Draw the two circular gradients
            dx = x - size // 2
            dy = y - size // 2
            dist = (dx * dx + dy * dy) ** 0.5
            t1 = dist / max_gradient_dist_1
            t2 = dist / max_gradient_dist_2
            p1 = 1 - t1
            p2 = 1 - t2
            value = value + p1 * gradient_weight_1 + p2 * gradient_weight_2
            # Convert to 0-255 range
            value = int(min(max(value * 255, 0), 255))
            blob_image.putpixel((x, y), value)

    # Enhance the contrast of the blob to make it nicer
    contrast = ImageEnhance.Contrast(blob_image)
    blob_image = contrast.enhance(10)

    return blob_image


def _seed_function(seed):
    """
    Return a number from the seed parameter.
    Ready to be used by random.seed, numpy.random.seed or torch.manual_seed.
    """
    if seed is None:
        return secrets.randbits(64)
    if isinstance(seed, str):
        # Use blake2b or blake2s algorithm to create a hash
        hash_object = hashlib.blake2b(
            seed.encode("utf-8"), digest_size=8, salt=b"blob-masks"
        )
        # Convert the hash digest to an integer
        return int.from_bytes(hash_object.digest(), "big")
    if isinstance(seed, (int, float)):
        return int(seed)

    raise ValueError("Invalid seed type. Seed must be a string or a number.")


# Create a wobbly effect to any image
def _wobbly_effect(
    image,
    amplitude_h,
    frequency_h,
    amplitude_v,
    frequency_v,
    phase_shift_h=0,
    phase_shift_v=0,
):
    width, height = image.size
    new_image = Image.new("RGBA", (width, height))

    for x in range(width):
        for y in range(height):
            dx = int(
                amplitude_h * math.sin(2 * math.pi * y / frequency_h + phase_shift_h)
            )
            dy = int(
                amplitude_v * math.sin(2 * math.pi * x / frequency_v + phase_shift_v)
            )
            new_x = x + dx
            new_y = y + dy

            if 0 <= new_x < width and 0 <= new_y < height:
                new_image.putpixel((x, y), image.getpixel((new_x, new_y)))
            else:
                new_image.putpixel((x, y), (0, 0, 0, 0))

    return new_image


def _wobbly_frame(seed=42):
    # Define the dimensions of the image and the size of the gradient
    size = 128
    width = size
    height = size
    gradient_size = 10

    # Create a new image with a black background
    image = Image.new("L", (width, height), "black")

    # Draw a white rectangle in the center of the image with a gradient
    draw = ImageDraw.Draw(image)
    center_rect = (
        gradient_size,
        gradient_size,
        width - gradient_size,
        height - gradient_size,
    )
    for i in range(gradient_size):
        gradient_color = int((255 * i) / gradient_size)
        draw.rectangle(center_rect, fill=gradient_color)
        center_rect = (
            center_rect[0] + 1,
            center_rect[1] + 1,
            center_rect[2] - 1,
            center_rect[3] - 1,
        )

    # Draw black borders around the image
    draw.rectangle((0, 0, width - 1, height - 1), outline="black")

    # Create some random wobbly effect
    random.seed(seed * 42)
    ah = random.randint(3, 5)
    fh = random.randint(50, 100)
    av = random.randint(3, 5)
    fv = random.randint(50, 100)
    sh = random.randint(1, 3)
    sv = random.randint(1, 3)
    image = _wobbly_effect(image, ah, fh, av, fv, math.pi / sh, math.pi / sv)

    # Add a blur to smooth the edges
    blur_radius = 5
    image = image.filter(ImageFilter.GaussianBlur(blur_radius))

    # Save the image
    return image


def blob_mask(seed=None, size=512):
    """
    Create a random blob mask.
    """

    seed = _seed_function(seed)
    # Multiply im and img pixel per pixel
    perlin_blob_image = _perlin_blob(seed).convert("L")
    wobbly_frame_image = _wobbly_frame(seed).convert("L")

    (width, height) = perlin_blob_image.size
    mix_image = Image.new("L", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = int(
                perlin_blob_image.getpixel((x, y))
                * wobbly_frame_image.getpixel((x, y))
                / 255
            )
            mix_image.putpixel((x, y), pixel)

    # Scale to 512x512
    new_size = (size, size)
    mix_image = mix_image.resize(new_size, Image.LANCZOS)

    # Enhance the contrast of the blob to make it nicer
    contrast = ImageEnhance.Contrast(mix_image)
    mix_image = contrast.enhance(10)

    # Convert the image to black and white (only, no greyscale)
    # if the pixel is above 0, it becomes white
    # if the pixel is 0, it becomes black
    def threshold(image):
        return image.point(lambda x: 0 if x == 0 else 255, mode="1")

    mix_image = threshold(mix_image)
    return mix_image


def borders_of_blob(blob_image, border_width=20):
    bw_image_np = np.array(blob_image.convert("L"))

    # Apply Canny edge detection
    edges = cv2.Canny(bw_image_np, 100, 200)
    image = edges[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    kernel = np.ones((border_width, border_width), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    borders_blob = Image.fromarray(dilated_edges)
    return borders_blob.convert("L")
