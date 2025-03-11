from pathlib import Path
import pandas as pd
from absl import app, flags
from cellot.utils.evaluate import compute_knn_enrichment
from cellot.losses.mmd import mmd_distance
import anndata as ad
from make_prediction import predict_from_unstim_data
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp

FLAGS = flags.FLAGS
flags.DEFINE_boolean("predictions", True, "Run predictions.")
flags.DEFINE_boolean("debug", False, "run in debug mode")
flags.DEFINE_string("outdir", "", "Path to outdir.")
flags.DEFINE_string("marker", "", "Marker to evaluate.")
flags.DEFINE_string("new_data_path", "", "Path to unseen data.")
flags.DEFINE_string(
    "n_cells", "100,250,500,1000,1500", "comma seperated list of integers"
)

flags.DEFINE_integer("n_reps", 10, "number of evaluation repetitions")
flags.DEFINE_string("embedding", None, "specify embedding context")
flags.DEFINE_string("evalprefix", None, "override default prefix")

flags.DEFINE_enum(
    "setting", "iid", ["iid", "ood"], "Evaluate iid, ood or via combinations."
)

flags.DEFINE_enum(
    "where",
    "data_space",
    ["data_space", "latent_space"],
    "In which space to conduct analysis",
)

flags.DEFINE_multi_string("via", "", "Directory containing compositional map.")

flags.DEFINE_string("subname", "", "")


def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])


def compute_pairwise_corrs(df):
    corr = df.corr().rename_axis(index="lhs", columns="rhs")
    return (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .set_index(["lhs", "rhs"])
        .squeeze()
        .rename()
    )


def compute_ks_test(treated, imputed):
    """Kolmogorov-Smirnov test"""
    treated = treated.values.flatten()
    imputed = imputed.values.flatten()
    ks_stat, ks_pval = ks_2samp(treated, imputed)
    return ks_stat, ks_pval


def compute_wasserstein(treated, imputed):
    """Wasserstein distance between distributions."""
    treated = treated.values.flatten()
    imputed = imputed.values.flatten()
    return wasserstein_distance(treated, imputed)


def compute_evaluations(iterator):
    gammas = np.logspace(1, -3, num=50)
    for ncells, nfeatures, treated, imputed in iterator:
        if len(treated) <= 1 or len(imputed) <= 1:
            print(
                f"Skipping evaluation: Not enough samples (treated={len(treated)}, imputed={len(imputed)})"
            )
            continue
        mut, mui = treated.mean(0), imputed.mean(0)
        stdt, stdi = treated.std(0), imputed.std(0)
        pwct = compute_pairwise_corrs(treated)
        pwci = compute_pairwise_corrs(imputed)

        yield ncells, nfeatures, "l2-means", np.linalg.norm(mut - mui)
        yield ncells, nfeatures, "l2-stds", np.linalg.norm(stdt - stdi)

        yield ncells, nfeatures, "l2-pairwise_feat_corrs", np.linalg.norm(pwct - pwci)

        wasserstein_dist = compute_wasserstein(treated, imputed)
        yield ncells, nfeatures, "wasserstein", wasserstein_dist

        ks_stat, ks_pval = compute_ks_test(treated, imputed)
        yield ncells, nfeatures, "ks-stat", ks_stat
        yield ncells, nfeatures, "ks-pval", ks_pval

        if treated.shape[1] < 1000:
            mmd = compute_mmd_loss(treated, imputed, gammas=gammas)
            yield ncells, nfeatures, "mmd", mmd

            knn, enrichment = compute_knn_enrichment(imputed, treated)
            k50 = enrichment.iloc[:, :50].values.mean()
            k100 = enrichment.iloc[:, :100].values.mean()

            yield ncells, nfeatures, "enrichment-k50", k50
            yield ncells, nfeatures, "enrichment-k100", k100


def main(argv):
    expdir = Path(FLAGS.outdir)
    new_data_path = Path(FLAGS.new_data_path)
    setting = FLAGS.setting
    where = FLAGS.where
    embedding = FLAGS.embedding
    prefix = FLAGS.evalprefix
    n_reps = FLAGS.n_reps
    marker = FLAGS.marker
    if (embedding is None) or len(embedding) == 0:
        embedding = None
    all_ncells = [int(x) for x in FLAGS.n_cells.split(",")]

    if prefix is None:
        prefix = f"evals_{setting}_{where}"
    outdir = expdir / prefix

    outdir.mkdir(exist_ok=True, parents=True)

    def iterate_feature_slices():
        unseen_data = ad.read(new_data_path)[:, marker]
        treated = unseen_data[unseen_data.obs["condition"] == "stim"]

        imputeddf = predict_from_unstim_data(expdir, new_data_path, "csv")
        imputeddf = imputeddf[[marker]]
        treateddf = treated.to_df()

        imputeddf.columns = imputeddf.columns.astype(str)
        treateddf.columns = treateddf.columns.astype(str)

        assert imputeddf.columns.equals(treateddf.columns)

        for ncells in all_ncells:
            max_cells = min(len(treateddf), len(imputeddf))
            if ncells > max_cells:
                print(f"Skipping ncells={ncells}: Not enough data (max={max_cells})")
                continue  # Évite un plantage si ncells > len(df)

            for r in range(n_reps):
                trt = treateddf.sample(ncells)
                imp = imputeddf.sample(ncells)
                yield ncells, "all", trt, imp

    evals = pd.DataFrame(
        compute_evaluations(iterate_feature_slices()),
        columns=["ncells", "nfeatures", "metric", "value"],
    )
    evals.to_csv(outdir / f"evals_{marker}.csv", index=None)

    return


if __name__ == "__main__":
    app.run(main)
