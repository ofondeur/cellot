import sys
from absl import app, flags
from pathlib import Path
import yaml
from make_prediction import predict_from_unstim_data

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "result_path", None, "Path to the trained model directory (e.g., ./results/...)"
)
flags.DEFINE_string(
    "unstim_data_path",
    None,
    "Path to the unstim data file (e.g., ./datasets/unstim_data.h5ad)",
)
flags.DEFINE_string(
    "output_path",
    None,
    "Path to save the predictions (e.g., ./predictions/output.h5ad)",
)


def main(argv):
    del argv
    if (
        FLAGS.result_path is None
        or FLAGS.unstim_data_path is None
        or FLAGS.output_path is None
    ):
        raise ValueError(
            "You need to give the 3 args --result_path, --unstim_data_path and --output_path"
        )

    result_path = Path(FLAGS.result_path).resolve()
    unstim_data_path = Path(FLAGS.unstim_data_path).resolve()
    output_path = Path(FLAGS.output_path).resolve()

    if not result_path.exists():
        raise FileNotFoundError(f"Result path does not exist : {result_path}")
    if not unstim_data_path.exists():
        raise FileNotFoundError(f"Data path does not exist : {unstim_data_path}")

    predict_from_unstim_data(result_path, unstim_data_path, output_path)

    print(f"Prediction made and saved under : {output_path}")


if __name__ == "__main__":
    app.run(main)
