from pathlib import Path
import pandas as pd
from absl import app, flags
from cellot.utils.evaluate import load_conditions
from cellot.utils import load_config
from cellot.data.cell import read_single_anndata
from scripts.evaluate import compute_evaluations
import anndata as ad
from make_prediction import predict_from_unstim_data

FLAGS = flags.FLAGS
flags.DEFINE_boolean("predictions", True, "Run predictions.")
flags.DEFINE_boolean("debug", False, "run in debug mode")
flags.DEFINE_string("outdir", "", "Path to outdir.")
flags.DEFINE_string("marker", "", "Marker to evaluate.")
flags.DEFINE_string("new_data_path", "", "Path to unseen data.")
flags.DEFINE_string("n_markers", None, "comma seperated list of integers")
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

    if FLAGS.n_markers is None:
        n_markers = None
    else:
        n_markers = FLAGS.n_markers.split(",")
    all_ncells = [int(x) for x in FLAGS.n_cells.split(",")]

    if prefix is None:
        prefix = f"evals_{setting}_{where}"
    outdir = expdir / prefix

    outdir.mkdir(exist_ok=True, parents=True)

    def iterate_feature_slices():

        assert (expdir / "config.yaml").exists()
        config = load_config(expdir / "config.yaml")
        if "ae_emb" in config.data:
            assert config.model.name == "cellot"
            config.data.ae_emb.path = str(expdir.parent / "model-scgen")

        cache = outdir / "imputed.h5ad"
        unseen_data = ad.read(new_data_path)[:, marker]
        unstim = unseen_data[unseen_data.obs["condition"] == "control"]
        imputed = unseen_data[unseen_data.obs["condition"] == "stim"]
        treateddf = predict_from_unstim_data(expdir, new_data_path, "csv")
        treateddf = treateddf.loc[:, marker]
        # _, treateddf, imputed = load_conditions(expdir, where, setting, embedding=embedding)

        imputed.write(cache)
        imputeddf = imputed.to_df()

        imputeddf.columns = imputeddf.columns.astype(str)
        treateddf.columns = treateddf.columns.astype(str)

        assert imputeddf.columns.equals(treateddf.columns)

        def load_markers():
            data = read_single_anndata(config, path=None)
            key = f"marker_genes-{config.data.condition}-rank"

            # rebuttal preprocessing stored marker genes using
            # a generic marker_genes-condition-rank key
            # instead of e.g. marker_genes-drug-rank
            # let's just patch that here:
            if key not in data.varm:
                key = "marker_genes-condition-rank"
                print("WARNING: using generic condition marker genes")

            sel_mg = data.varm[key][config.data.target].sort_values().index
            return sel_mg

        if n_markers is not None:
            markers = load_markers()
            for k in n_markers:
                if k != "all":
                    feats = markers[: int(k)]
                else:
                    feats = list(markers)

                for ncells in all_ncells:
                    if ncells > min(len(treateddf), len(imputeddf)):
                        break
                    for r in range(n_reps):
                        trt = treateddf[feats].sample(ncells)
                        imp = imputeddf[feats].sample(ncells)
                        yield ncells, k, trt, imp

        else:
            for ncells in all_ncells:
                if ncells > min(len(treateddf), len(imputeddf)):
                    break
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
