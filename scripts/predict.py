from absl import app, flags
from pathlib import Path
from make_prediction import predict_from_unstim_data
from pathlib import Path
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "result_path", None, "Path to the trained model directory (e.g., ./results/...)"
)
flags.DEFINE_string(
    "unstim_data_path",
    None,
    "Path to the unstim data file (e.g., ./datasets/unstim_data.h5ad)",
)
flags.DEFINE_string("output_path", None, "Path to save the predictions (optional)")
flags.DEFINE_string(
    "pred_format", "h5ad", "Format for saving the predictions (csv or h5ad)"
)


def main(argv):
    del argv

    if not FLAGS.result_path or not FLAGS.unstim_data_path:
        raise ValueError("You need to specify --result_path and --unstim_data_path")

    result_path = Path(FLAGS.result_path).resolve()
    unstim_data_path = Path(FLAGS.unstim_data_path).resolve()
    output_path = Path(FLAGS.output_path).resolve() if FLAGS.output_path else None

    if not result_path.exists():
        raise FileNotFoundError(f"Result path does not exist: {result_path}")
    if not unstim_data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {unstim_data_path}")

    prediction = predict_from_unstim_data(
        result_path, unstim_data_path, FLAGS.pred_format, output_path
    )

    if output_path:
        print(f"Prediction saved under: {output_path}")
    else:
        print("Prediction completed but not saved (no output path specified)")


if __name__ == "__main__":
    app.run(main)
