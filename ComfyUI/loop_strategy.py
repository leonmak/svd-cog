from PIL import Image, ImageSequence

def loop(input_path, plays=0):
    original_gif = Image.open(input_path)
    original_gif.save(
        input_path,
        save_all=True,
        append_images=[original_gif],
        loop=plays,
        duration=original_gif.info.get('duration', 1000)
    )

def loop_reverse(input_path):
    original_gif = Image.open(input_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(original_gif)]
    back = frames[::-1]
    frames.extend(back)
    frames[0].save(input_path, save_all=True, loop=0, append_images=frames[1:])
