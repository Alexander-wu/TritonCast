### 1. Download the Dataset

First, you need to download the dataset required for the project.

- **Download Link**: [McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt](https://huggingface.co/datasets/scaomath/navier-stokes-dataset/blob/main/McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt)

After downloading, place the dataset file `McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt` into the `dataset` folder within the project's root directory. If the folder does not exist, please create it.

Your directory structure should look like this:

```
EXP_2D_ISOTROPIC_TURBULENCE/
|-- dataset/
|   |-- McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt
|-- ... (other folders and files)
```

### 2. Download Pre-trained Weights

Next, download the pre-trained model weights. These include weights for TritonCast and other baseline models (e.g., FNO, SimVP).

- **Download Link**: [TritonCast & Baselines Checkpoints](https://huggingface.co/TritonCast/TritonCast_model/tree/main/exp_2d_isotropic_turbulence/checkpoints)

Please download all model files from the `checkpoints` directory and place them into the `checkpoints` folder in your project's root directory.

Your directory structure should look like this:

```
EXP_2D_ISOTROPIC_TURBULENCE/
|-- checkpoints/
|   |-- FNO/
|   |   |-- model.pt
|   |-- SimVP/
|   |   |-- model.pt
|   |-- Triton/
|   |   |-- model.pt
|   |-- ... (other model folders)
|-- ... (other folders and files)
```

### 3. Configure and Run Inference

Once everything is set up, you can select the model for inference by modifying the configuration file.

1.  Open the `config.yaml` file located in the project's root directory.
2.  Find the `selected_model` field.
3.  Change its value to the name of the model you wish to test. The available model names correspond to the subfolder names in the `checkpoints` directory, such as `'Triton'`, `'SimVP'`, `'FNO'`, etc.

For example, if you want to run inference with the **SimVP** model, your `config.yaml` should look like this:

```yaml
selected_model: 'SimVP'

models:
  # ... (model parameters)
```

After modifying the file, save it and run the inference script:

```bash
python inference.py
```