import os
import anndata as ad
from cellot.utils.helpers import load_config
from cellot.utils.loaders import load
from cellot.data.cell import read_list
from cellot.data.cell import AnnDataDataset
from torch.utils.data import DataLoader


def predict_from_unstim_data(result_path, unstim_data_path, output_path):
    config_path = os.path.join(result_path, "config.yaml")
    chkpt = os.path.join(result_path, "cache/model.pt")

    # load the config and then the model (f,g)
    config = load_config(config_path)
    (_, g), _, _ = load(config, restore=chkpt)
    g.eval()

    # load the data to predict and filter with the interzsting markers
    unstim_anndata_to_predict = ad.read(unstim_data_path)
    features = read_list(config.data.features)
    unstim_anndata_to_predict = unstim_anndata_to_predict[:, features].copy()

    # predict the data (first put it in the dataset format)
    dataset_args = {}
    dataset = AnnDataDataset(unstim_anndata_to_predict.copy(), **dataset_args)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))

    outputs = g.transport(inputs.requires_grad_(True)).detach().numpy()
    predicted = ad.AnnData(
        outputs,
        obs=dataset.adata.obs.copy(),
        var=dataset.adata.var.copy(),
    )

    # save the prediction in the desired format
    if output_path.endswith(".csv"):
        prediction = predicted.to_df()
        prediction.to_csv(output_path)

    elif output_path.endswith(".h5ad"):
        predicted.write(output_path)
    return


# tests
result_path = "/Users/MacBook/stanford/cellot/results/test_new/model-cellot"
unstim_data_path = (
    "/Users/MacBook/stanford/cellot/datasets/atest_data/unstim_Bcell_to_predict.h5ad"
)
output_path = (
    "/Users/MacBook/stanford/cellot/results/test_new/unseen_data_result/prediction.h5ad"
)
predict_from_unstim_data(result_path, unstim_data_path, output_path)
