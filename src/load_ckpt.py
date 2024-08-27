import hashlib
import torch
from mineclip import MineCLIP


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (
        hashlib.md5(open("./attn.pth", "rb").read()).hexdigest() == "b5ece9198337cfd117a3bfbd921e56da"
    ), "broken ckpt"

    model = MineCLIP("vit_base_p16_fz.v2.t2", hidden_dim=512, image_feature_dim=512, mlp_adapter_spec="v0-2.t0", pool_type="attn.d2.nh8.glusw", resolution=[160, 256]).to(device)
    model.load_ckpt("./attn.pth", strict=True)
    print("Successfully loaded ckpt")


if __name__ == "__main__":
    main()
