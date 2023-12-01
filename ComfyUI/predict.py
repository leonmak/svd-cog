import os
import random
import sys
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
    VAEDecode,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    KSampler,
    EmptyLatentImage,
    NODE_CLASS_MAPPINGS,
)


def run(img_text, iterations):
    if not img_text:
        raise Exception('No Input')
    
    import_custom_nodes()
    
    with torch.inference_mode():
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_20 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_turbo_1.0_fp16.safetensors"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text=img_text,
            clip=get_value_at_index(checkpointloadersimple_20, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="text, watermark",
            clip=get_value_at_index(checkpointloadersimple_20, 1),
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_14 = ksamplerselect.get_sampler(sampler_name="euler_ancestral")

        # seed_rgthree = NODE_CLASS_MAPPINGS["Seed (rgthree)"]()
        # seed_rgthree_28 = seed_rgthree.main(seed=random.randint(1, 2**64))

        imageonlycheckpointloader = NODE_CLASS_MAPPINGS["ImageOnlyCheckpointLoader"]()
        imageonlycheckpointloader_29 = imageonlycheckpointloader.load_checkpoint(
            ckpt_name="svd_xt.safetensors"
        )

        # freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        # freeu_v2_31 = freeu_v2.patch(b1=1.3, b2=1.4, s1=0.9, s2=0.2)

        sdturboscheduler = NODE_CLASS_MAPPINGS["SDTurboScheduler"]()
        sdturboscheduler_22 = sdturboscheduler.get_sigmas(
            steps=1, model=get_value_at_index(checkpointloadersimple_20, 0)
        )

        samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        samplercustom_13 = samplercustom.sample(
            add_noise=True,
            noise_seed=random.randint(1, 2**64),
            cfg=1,
            model=get_value_at_index(checkpointloadersimple_20, 0),
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            sampler=get_value_at_index(ksamplerselect_14, 0),
            sigmas=get_value_at_index(sdturboscheduler_22, 0),
            latent_image=get_value_at_index(emptylatentimage_5, 0),
        )

        vaedecode = VAEDecode()
        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(samplercustom_13, 0),
            vae=get_value_at_index(checkpointloadersimple_20, 2),
        )

        svd_img2vid_conditioning = NODE_CLASS_MAPPINGS["SVD_img2vid_Conditioning"]()
        svd_img2vid_conditioning_33 = svd_img2vid_conditioning.encode(
            width=512,
            height=512,
            video_frames=25,
            motion_bucket_id=40,
            fps=12,
            augmentation_level=0.02,
            clip_vision=get_value_at_index(imageonlycheckpointloader_29, 1),
            init_image=get_value_at_index(vaedecode_8, 0),
            vae=get_value_at_index(imageonlycheckpointloader_29, 2),
        )

        videolinearcfgguidance = NODE_CLASS_MAPPINGS["VideoLinearCFGGuidance"]()
        ksampler = KSampler()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(iterations):
            videolinearcfgguidance_30 = videolinearcfgguidance.patch(
                min_cfg=1, model=get_value_at_index(imageonlycheckpointloader_29, 0)
            )

            ksampler_35 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=2.5,
                sampler_name="euler",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(videolinearcfgguidance_30, 0),
                positive=get_value_at_index(svd_img2vid_conditioning_33, 0),
                negative=get_value_at_index(svd_img2vid_conditioning_33, 1),
                latent_image=get_value_at_index(svd_img2vid_conditioning_33, 2),
            )

            vaedecode_34 = vaedecode.decode(
                samples=get_value_at_index(ksampler_35, 0),
                vae=get_value_at_index(imageonlycheckpointloader_29, 2),
            )

            vhs_videocombine_36 = vhs_videocombine.combine_video(
                frame_rate=20,
                loop_count=0,
                filename_prefix="SDXL-Turbo-SDV",
                format="image/gif",
                pingpong=False,
                save_image=True,
                crf=20,
                save_metadata=False,
                audio_file="",
                images=get_value_at_index(vaedecode_34, 0),
            )
    
    return vhs_videocombine_36['ui']['gifs'][0]


from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        image_text: str = Input(description="Input image text"),
        # image: Path = Input(description="Input image"),
        iterations: int = Input(description="iterations", default=3),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
    ) -> Path:
        """Run a single prediction on the model"""
        # image_path_str = image.absolute().as_posix() if image else ''
        output_dir = 'ComfyUI/output'
        os.makedirs(output_dir, exist_ok=True)
        for file_name in os.listdir(output_dir):
            os.remove(Path(output_dir, file_name))
        res = run(image_text, iterations)
        return Path(output_dir, res['filename'])
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
