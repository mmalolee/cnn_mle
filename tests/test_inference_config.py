import pytest
from src.cfg.inference_config import InferenceConfig
from dataclasses import FrozenInstanceError
from torchvision import transforms


# # --- FIXTURES -------------------------------------
@pytest.fixture(scope="function")
def inference_config_with_default():
    return InferenceConfig(device="cuda")


@pytest.fixture(scope="function")
def inference_config_with_values():
    return InferenceConfig(
        device="cpu",
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
    )


# # --- TESTS ----------------------------------------
@pytest.mark.parametrize(
    "attr, value",
    [
        ("device", "cuda"),
        ("mean", (0.5, 0.5, 0.5)),
        ("std", (0.5, 0.5, 0.5)),
    ],
)
def test_with_default_values(inference_config_with_default, attr, value):
    assert getattr(inference_config_with_default, attr) == value


@pytest.mark.parametrize(
    "attr, value",
    [
        ("device", "cpu"),
        ("mean", (0.1, 0.2, 0.3)),
        ("std", (0.4, 0.5, 0.6)),
    ],
)
def test_with_values(inference_config_with_values, attr, value):
    assert getattr(inference_config_with_values, attr) == value


def test_is_in_allowed_devices(inference_config_with_values):
    assert inference_config_with_values.device in InferenceConfig.ALLOWED_DEVICES


@pytest.mark.parametrize(
    "attr, instance",
    [
        ("device", str),
        ("mean", tuple),
        ("std", tuple),
    ],
)
def test_instances(inference_config_with_values, attr, instance):
    assert isinstance(getattr(inference_config_with_values, attr), instance)


def test_tuple_content_instances(inference_config_with_values):
    assert all(isinstance(m, float) for m in inference_config_with_values.mean)
    assert all(isinstance(s, float) for s in inference_config_with_values.std)


def test_img_size_value(inference_config_with_default):
    assert inference_config_with_default.img_size == 250


def test_img_size_instance(inference_config_with_default):
    assert isinstance(inference_config_with_default.img_size, int)


def test_allowed_devices_instances():
    assert isinstance(InferenceConfig.ALLOWED_DEVICES, tuple)
    assert all(isinstance(d, str) for d in InferenceConfig.ALLOWED_DEVICES)


@pytest.mark.parametrize(
    "transformer, instance",
    [
        ("pil_transforms", list),
        ("tensor_transforms", list),
        ("basic_transformer", transforms.Compose),
    ],
)
def test_transformers_instances(inference_config_with_values, transformer, instance):
    assert isinstance(getattr(inference_config_with_values, transformer), instance)


def test_is_frozen(inference_config_with_values):
    with pytest.raises(FrozenInstanceError):
        setattr(inference_config_with_values, "device", "pegasus")


def test_invalid_device_error():
    with pytest.raises(ValueError):
        InferenceConfig(device="pegasus")


def test_transformer_order(inference_config_with_values):
    expected_order = [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    types = [type(t) for t in inference_config_with_values.basic_transformer.transforms]

    assert types == expected_order


def test_resize_value(inference_config_with_default):
    resizer = inference_config_with_default.pil_transforms[0]
    size = resizer.size
    assert size == [
        inference_config_with_default.img_size,
        inference_config_with_default.img_size,
    ]


@pytest.mark.parametrize("bad_mean_len", [(0.1, 0.2), (0.1, 0.2, 0.3, 0.4)])
def test_mean_len(bad_mean_len):
    with pytest.raises(ValueError):
        InferenceConfig(
            device="cuda",
            mean=bad_mean_len,
            std=(0.1, 0.1, 0.1),
        )


@pytest.mark.parametrize("bad_std_len", [(0.1, 0.2), (0.1, 0.2, 0.3, 0.4)])
def test_std_len(bad_std_len):
    with pytest.raises(ValueError):
        InferenceConfig(
            device="cuda",
            mean=(0.1, 0.1, 0.1),
            std=bad_std_len,
        )


def test_is_std_zero():
    with pytest.raises(ValueError):
        InferenceConfig(
            device="cuda",
            mean=(0.1, 0.1, 0.1),
            std=(0.1, 0, 0.1),
        )
