# External libraries

To use a local editable clone of Hugging Face `diffusers` inside this repository, clone it to:

```bash
git clone https://github.com/huggingface/diffusers.git external/diffusers
```

The new scripts automatically prefer:

- `--diffusers_src <path>`
- `$DIFFUSERS_SRC`
- `external/diffusers/src` (default)

So once cloned, no extra changes are needed.
