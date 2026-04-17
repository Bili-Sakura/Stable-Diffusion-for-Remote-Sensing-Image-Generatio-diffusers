import argparse
from pathlib import Path

from scripts._diffusers_import import setup_diffusers_import_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert original SD checkpoint to native diffusers pipeline.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .ckpt or .safetensors file.")
    parser.add_argument("--output_path", type=str, required=True, help="Output diffusers model directory.")
    parser.add_argument("--original_config_file", type=str, default=None, help="Original LDM config yaml path.")
    parser.add_argument("--extract_ema", action="store_true", help="Extract EMA weights when present.")
    parser.add_argument("--from_safetensors", action="store_true", help="Set if checkpoint is in safetensors format.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_in_channels", type=int, default=None)
    parser.add_argument("--scheduler_type", type=str, default="pndm", choices=["pndm", "ddim", "euler", "euler-ancestral"])
    parser.add_argument("--prediction_type", type=str, default=None, choices=["epsilon", "v_prediction"])
    parser.add_argument("--upcast_attention", action="store_true")
    parser.add_argument("--load_safety_checker", action="store_true")
    parser.add_argument("--diffusers_src", type=str, default=None, help="Local diffusers/src path.")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_diffusers_import_path(args.diffusers_src)

    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        download_from_original_stable_diffusion_ckpt,
    )

    pipeline = download_from_original_stable_diffusion_ckpt(
        checkpoint_path=args.checkpoint_path,
        original_config_file=args.original_config_file,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        extract_ema=args.extract_ema,
        num_in_channels=args.num_in_channels,
        scheduler_type=args.scheduler_type,
        upcast_attention=args.upcast_attention,
        from_safetensors=args.from_safetensors,
        load_safety_checker=args.load_safety_checker,
    )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(output_path)
    print(f"Saved converted diffusers pipeline to {output_path}")


if __name__ == "__main__":
    main()
