import argparse
from pathlib import Path

import torch

from scripts._diffusers_import import setup_diffusers_import_path


def parse_args():
    parser = argparse.ArgumentParser(description="Native diffusers text-to-image inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Diffusers model id or local path.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="outputs/diffusers")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffusers_src", type=str, default=None, help="Local diffusers/src path.")
    return parser.parse_args()


def _dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()
    setup_diffusers_import_path(args.diffusers_src)

    from diffusers import StableDiffusionPipeline

    dtype = _dtype_from_name(args.torch_dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe = pipe.to(device)

    generator = torch.Generator(device=str(device)).manual_seed(args.seed)
    output = pipe(
        prompt=[args.prompt] * args.num_images,
        negative_prompt=[args.negative_prompt] * args.num_images if args.negative_prompt else None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(output.images):
        image.save(outdir / f"sample_{idx:03d}.png")

    print(f"Saved {len(output.images)} images to {outdir}")


if __name__ == "__main__":
    main()
