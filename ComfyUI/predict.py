import sys
from cog import BasePredictor, Input, Path
import os
import random
from typing import Sequence, Mapping, Any, Union
import torch

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()



from nodes import (
    NODE_CLASS_MAPPINGS,
    LoadImage,
    VAEEncodeForInpaint,
    LatentComposite,
    KSampler,
    VAEDecode,
    VAEEncode,
    ImageScale,
)
def run(image_path, width, height, frames, fps,
        motion_bucket_id, cond_aug,
        ksampler_steps, cfg, crf,
        mask_radius, grow_mask_by, output_format):
    import_custom_nodes()

    with torch.inference_mode():
        imageonlycheckpointloader = NODE_CLASS_MAPPINGS["ImageOnlyCheckpointLoader"](
        )
        imageonlycheckpointloader_15 = imageonlycheckpointloader.load_checkpoint(
            ckpt_name="svd-fp16.safetensors"
        )

        loadimage = LoadImage()
        loadimage_23 = loadimage.load_image(
            image=image_path
        )

        svd_img2vid_conditioning = NODE_CLASS_MAPPINGS["SVD_img2vid_Conditioning"](
        )
        svd_img2vid_conditioning_12 = svd_img2vid_conditioning.encode(
            width=width,
            height=height,
            video_frames=frames,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            augmentation_level=cond_aug,
            clip_vision=get_value_at_index(imageonlycheckpointloader_15, 1),
            init_image=get_value_at_index(loadimage_23, 0),
            vae=get_value_at_index(imageonlycheckpointloader_15, 2),
        )

        imagescale = ImageScale()
        imagescale_34 = imagescale.upscale(
            upscale_method="nearest-exact",
            width=width,
            height=height,
            crop="disabled",
            image=get_value_at_index(loadimage_23, 0),
        )

        mask_gaussian_region = NODE_CLASS_MAPPINGS["Mask Gaussian Region"]()
        mask_gaussian_region_41 = mask_gaussian_region.gaussian_region(
            radius=mask_radius, masks=get_value_at_index(loadimage_23, 1)
        )

        vaeencodeforinpaint = VAEEncodeForInpaint()
        vaeencodeforinpaint_27 = vaeencodeforinpaint.encode(
            grow_mask_by=grow_mask_by,
            pixels=get_value_at_index(imagescale_34, 0),
            vae=get_value_at_index(imageonlycheckpointloader_15, 2),
            mask=get_value_at_index(mask_gaussian_region_41, 0),
        )

        vaeencode = VAEEncode()
        vaeencode_30 = vaeencode.encode(
            pixels=get_value_at_index(imagescale_34, 0),
            vae=get_value_at_index(imageonlycheckpointloader_15, 2),
        )

        videolinearcfgguidance = NODE_CLASS_MAPPINGS["VideoLinearCFGGuidance"](
        )
        vhs_duplicatelatents = NODE_CLASS_MAPPINGS["VHS_DuplicateLatents"]()
        latentcomposite = LatentComposite()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        videolinearcfgguidance_14 = videolinearcfgguidance.patch(
            min_cfg=1, model=get_value_at_index(imageonlycheckpointloader_15, 0)
        )

        vhs_duplicatelatents_28 = vhs_duplicatelatents.duplicate_input(
            multiply_by=14, latents=get_value_at_index(vaeencodeforinpaint_27, 0)
        )

        vhs_duplicatelatents_31 = vhs_duplicatelatents.duplicate_input(
            multiply_by=14, latents=get_value_at_index(vaeencode_30, 0)
        )

        latentcomposite_33 = latentcomposite.composite(
            x=0,
            y=0,
            feather=0,
            samples_to=get_value_at_index(vhs_duplicatelatents_28, 0),
            samples_from=get_value_at_index(vhs_duplicatelatents_31, 0),
        )

        ksampler_3 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=ksampler_steps,
            cfg=cfg,
            sampler_name="EulerEDMSampler",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(videolinearcfgguidance_14, 0),
            positive=get_value_at_index(svd_img2vid_conditioning_12, 0),
            negative=get_value_at_index(svd_img2vid_conditioning_12, 1),
            latent_image=get_value_at_index(latentcomposite_33, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(imageonlycheckpointloader_15, 2),
        )

        vhs_videocombine_42 = vhs_videocombine.combine_video(
            frame_rate=8,
            loop_count=0,
            filename_prefix="animate",
            format=output_format,
            pingpong=False,
            save_image=True,
            crf=crf,
            save_metadata=True,
            audio_file="",
            images=get_value_at_index(vaedecode_8, 0),
        )
        return vhs_videocombine_42['ui']['gifs'][0]


from sizing_strategy import SizingStrategy

class Predictor(BasePredictor):    
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.sizing_strategy = SizingStrategy()

    def predict(
        self,
        image_path: Path = Input(description="Input image"),
        fps: int = Input(description="Frames per second", default=14),
        frames: int = Input(description="Frames", default=28),
        motion_bucket_id: int = Input(
            description="overall motion", default=127, ge=1, le=255),
        cond_aug: float = Input(description="noise",
                                default=0.02, ge=-0.4, le=0.4),
        ksampler_steps: int = Input(
            description="more accurate to prompt but longer", default=20, ge=1, le=90),
        cfg: float = Input(description="cfg", default=2.5, ge=0, le=10),
        crf: int = Input(description="crf", default=20, ge=0, le=100),
        mask_radius: float = Input(
            description="radius of mask", default=10.1, ge=1, le=50),
        grow_mask_by: int = Input(
            description="grow mask by", default=6, ge=0, le=50),
        file_format: str = Input(description="output file format",
                                 choices=[
                                     'image/gif', 'image/webp', 'video/h264-mp4', 'video/h265-mp4', 'video/webm'],
                                 default='image/gif'),
        sizing_strategy: str = Input(
            description="Decide how to resize the input image",
            choices=[
                "maintain_aspect_ratio",
                "crop_to_16_9",
                "use_image_dimensions",
            ],
            default="maintain_aspect_ratio",
        ),
        # playback: str = Input(
        #     description="Decide how to resize the input image",
        #     choices=[
        #         "loop",
        #         "once",
        #         "reverse loop",
        #     ],
        #     default="loop",
        # ),
    ) -> Path:
        """Run a single prediction on the model"""
        # clear image
        output_dir = 'ComfyUI/output'
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(output_dir):
            os.remove(Path(output_dir, file_name))

        # resize and save image
        (image, width, height) = self.sizing_strategy.apply(
            image=str(image_path),
            sizing_strategy=sizing_strategy)
        image.save(image_path)

        res = run(str(image_path),
                  width, height,
                  frames, fps,
                  motion_bucket_id, cond_aug,
                  ksampler_steps, cfg, crf,
                  mask_radius, grow_mask_by,
                  file_format)

        output_path = Path(output_dir, res['filename'])

        # if playback == 'loop':
        #     loop(output_path)
        # elif playback == 'once':
        #     loop(output_path, plays=1)
        # elif playback == 'reverse loop':
        #     loop(output_path, reverse=True)

        return output_path
