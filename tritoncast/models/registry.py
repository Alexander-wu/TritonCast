import importlib


_COMMON_SINGLE_CHANNEL_ARGS = {
    "n_input_scalar_components": 1,
    "n_input_vector_components": 0,
    "n_output_scalar_components": 1,
    "n_output_vector_components": 0,
    "time_history": 1,
    "time_future": 1,
    "activation": "silu",
}

_FOURIER_UNET_ARGS = {
    **_COMMON_SINGLE_CHANNEL_ARGS,
    "hidden_channels": 32,
    "modes1": 8,
    "modes2": 8,
    "norm": True,
    "n_fourier_layers": 1,
}

def _load_attr(module_path, attr_name):
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _factory(module_path, attr_name):
    def _build(**kwargs):
        return _load_attr(module_path, attr_name)(**kwargs)

    return _build


def _factory_with_fixed_kwargs(module_path, attr_name, fixed_kwargs):
    def _build(**kwargs):
        model_cls = _load_attr(module_path, attr_name)
        return model_cls(**{**fixed_kwargs, **kwargs})

    return _build


_REGISTRY = {
    ("medium_range_weather", "Triton"): _factory("tritoncast.models.triton_weather", "Triton"),
    ("long_term_stability", "Triton"): _factory("tritoncast.models.triton_rollout", "Triton"),
    ("long_term_stability", "Pangu"): _factory(
        "experiments.exp2_long_term_stability_test.model_baselines.pangu_model", "Pangu"
    ),
    ("long_term_stability", "Fuxi"): _factory(
        "experiments.exp2_long_term_stability_test.model_baselines.fuxi_model", "Fuxi"
    ),
    ("multi_year_climate", "Triton"): _factory("tritoncast.models.triton_weather", "Triton"),
    ("global_ocean_forecasting", "Triton"): _factory(
        "experiments.exp4_global_ocean_simulation_and_forecasting.networks.Triton", "Triton"
    ),
    ("global_ocean_forecasting", "TritonAtmos"): _factory(
        "experiments.exp4_global_ocean_simulation_and_forecasting.networks.Triton_atmos", "Triton"
    ),
    ("global_ocean_simulation", "Triton"): _factory(
        "experiments.exp4_global_ocean_simulation_and_forecasting.networks.Triton", "Triton"
    ),
    ("isotropic_turbulence", "Triton"): _factory(
        "experiments.exp7_isotropic_turbulence.model.Triton_turbulence_model", "Triton_Turbulence"
    ),
    ("isotropic_turbulence", "Triton_V2"): _factory(
        "experiments.exp7_isotropic_turbulence.model.triton_model_v2", "Triton_v2"
    ),
    ("isotropic_turbulence", "FNO"): _factory("experiments.exp7_isotropic_turbulence.model_baselines.fno", "FNO2d"),
    ("isotropic_turbulence", "DiT"): _factory("experiments.exp7_isotropic_turbulence.model_baselines.dit", "Dit"),
    ("isotropic_turbulence", "SimVP"): _factory("experiments.exp7_isotropic_turbulence.model_baselines.simvp", "SimVP"),
    ("isotropic_turbulence", "CNO"): _factory("experiments.exp7_isotropic_turbulence.model_baselines.cno", "CNO"),
    ("isotropic_turbulence", "MGNO"): _factory("experiments.exp7_isotropic_turbulence.model_baselines.mgno", "MgNO"),
    ("isotropic_turbulence", "LSM"): _factory("experiments.exp7_isotropic_turbulence.model_baselines.lsm", "LSM"),
    ("isotropic_turbulence", "PastNet"): _factory(
        "experiments.exp7_isotropic_turbulence.model_baselines.pastnet", "PastNetModel"
    ),
    ("isotropic_turbulence", "ResNet"): _factory(
        "experiments.exp7_isotropic_turbulence.model_baselines.resnet", "ResNet"
    ),
    ("isotropic_turbulence", "U_net"): _factory("experiments.exp7_isotropic_turbulence.model_baselines.unet", "U_net"),
    ("isotropic_turbulence", "FourierUnet"): _factory_with_fixed_kwargs(
        "experiments.exp7_isotropic_turbulence.model_baselines.fourier_unet",
        "FourierUnet",
        _FOURIER_UNET_ARGS,
    ),
}


def build_model(model_name, experiment, **kwargs):
    try:
        builder = _REGISTRY[(experiment, model_name)]
    except KeyError as exc:
        raise KeyError(f"Unsupported model '{model_name}' for experiment '{experiment}'.") from exc

    return builder(**kwargs)

