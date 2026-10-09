import argparse

import torch
import torchvision.transforms as T
from PIL import Image

from vpr_model import VPRModel


def load_checkpoint(path):
    print(f"[1] Loading checkpoint: {path}")

    ckpt = torch.load(path, map_location="cpu")

    # 兼容普通 state_dict 和 Lightning checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    return state_dict


def infer_salad_config(state_dict):
    """
    根据 checkpoint 中各层权重尺寸自动判断 SALAD 配置。
    """

    token_key = "aggregator.token_features.2.weight"
    cluster_key = "aggregator.cluster_features.3.weight"
    score_key = "aggregator.score.3.weight"

    if token_key not in state_dict:
        raise RuntimeError(
            "无法从 checkpoint 找到 SALAD 权重。\n"
            f"缺少 key: {token_key}\n"
            "可以先打印 state_dict.keys() 检查 checkpoint 格式。"
        )

    token_dim = state_dict[token_key].shape[0]
    cluster_dim = state_dict[cluster_key].shape[0]
    num_clusters = state_dict[score_key].shape[0]

    descriptor_dim = num_clusters * cluster_dim + token_dim

    print("[2] Detected SALAD config:")
    print(f"    num_clusters = {num_clusters}")
    print(f"    cluster_dim  = {cluster_dim}")
    print(f"    token_dim    = {token_dim}")
    print(f"    descriptor   = {descriptor_dim}")

    return {
        "num_channels": 768,
        "num_clusters": num_clusters,
        "cluster_dim": cluster_dim,
        "token_dim": token_dim,
    }


def build_model(state_dict, device):
    agg_config = infer_salad_config(state_dict)

    print("[3] Building DINOv2 + SALAD model...")

    model = VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        agg_arch="SALAD",
        agg_config=agg_config,
    )

    print("[4] Loading SALAD weights...")

    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    return model


def get_transform():
    return T.Compose([
        T.Resize(
            (322, 322),
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_image(path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        required=True,
        help="SALAD checkpoint path"
    )

    parser.add_argument(
        "--image1",
        default=None
    )

    parser.add_argument(
        "--image2",
        default=None
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    state_dict = load_checkpoint(args.ckpt)
    model = build_model(state_dict, device)

    # --------------------------------------------------
    # 如果没有给图片，先用随机 tensor 测试 forward
    # --------------------------------------------------

    if args.image1 is None:
        print("[5] Running dummy forward test...")

        x = torch.randn(
            2, 3, 322, 322,
            device=device
        )

        with torch.inference_mode():
            descriptors = model(x)

        print()
        print("===== SUCCESS =====")
        print("Descriptor shape:")
        print(descriptors.shape)

        print("Descriptor L2 norm:")
        print(torch.linalg.norm(descriptors, dim=1))

        return

    # --------------------------------------------------
    # 实际图片测试
    # --------------------------------------------------

    transform = get_transform()

    img1 = load_image(args.image1, transform)

    images = [img1]

    if args.image2 is not None:
        img2 = load_image(args.image2, transform)
        images.append(img2)

    batch = torch.stack(images).to(device)

    print("[5] Extracting descriptors...")

    with torch.inference_mode():

        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                descriptors = model(batch)
        else:
            descriptors = model(batch)

    descriptors = descriptors.float().cpu()

    print()
    print("===== RESULT =====")

    print("Descriptor shape:")
    print(descriptors.shape)

    print()

    print("Descriptor L2 norm:")
    print(torch.linalg.norm(descriptors, dim=1))

    # 两张图片就比较相似度
    if len(descriptors) == 2:

        d1 = descriptors[0]
        d2 = descriptors[1]

        # SALAD 输出本身已经做了 L2 normalize
        cosine_similarity = torch.dot(d1, d2).item()

        l2_distance = torch.linalg.norm(
            d1 - d2
        ).item()

        print()
        print("Cosine similarity:")
        print(cosine_similarity)

        print()
        print("L2 distance:")
        print(l2_distance)


if __name__ == "__main__":
    main()
