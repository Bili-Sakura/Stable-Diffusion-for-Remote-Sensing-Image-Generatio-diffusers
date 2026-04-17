import argparse
from pathlib import Path

from scripts._diffusers_import import setup_diffusers_import_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RSITMD LDM checkpoint to diffusers format.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to RSITMD .ckpt/.safetensors.")
    parser.add_argument("--output_path", type=str, required=True, help="Output diffusers model directory.")
    parser.add_argument(
        "--original_config_file",
        type=str,
        default="configs/stable-diffusion/RSITMD.yaml",
        help="Original RSITMD LDM config yaml.",
    )
    parser.add_argument("--extract_ema", action="store_true")
    parser.add_argument("--from_safetensors", action="store_true")
    parser.add_argument("--scheduler_type", type=str, default="pndm", choices=["pndm", "ddim", "euler", "euler-ancestral"])
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
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        from_safetensors=args.from_safetensors,
    )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(output_path)
    print(f"Saved converted RSITMD diffusers pipeline to {output_path}")


if __name__ == "__main__":
    main()
