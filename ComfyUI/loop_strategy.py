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
    
    # Create a new list of frames with original frames followed by reversed frames for looping
    frames = [frame.copy() for frame in ImageSequence.Iterator(original_gif)]
    forward = frames[::]
    frames.reverse()
    forward.extend(frames)
    frames[0].save(input_path, save_all=True, append_images=forward[1:])
