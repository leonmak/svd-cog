from PIL import Image, ImageSequence


def loop(input_path, plays=0, reverse=False):
    original_gif = Image.open(input_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(original_gif)]
    if reverse:
        back = frames[::-1]
        frames.extend(back)
    frames[0].save(input_path,
                   save_all=True,
                   loop=plays,
                   append_images=frames[1:])
