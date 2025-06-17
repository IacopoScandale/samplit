import os
from typing import Any

import pandas as pd
from jiwer import wer

from data.strings import J_EVALUATION_DIR
from data.utils import load_jamendo_dataframe


def iou(row):
  inter_start = max(row["start_time"], row["model_start_time"])
  union_start = min(row["start_time"], row["model_start_time"])
  inter_end = min(row["end_time"], row["model_end_time"])
  union_end = max(row["end_time"], row["model_end_time"])

  inter = max(0, inter_end - inter_start)
  union = union_end - union_start
  return inter / union if union > 0 else 0


def get_models_df(num_words: int, language: str | None = None) -> pd.DataFrame:
  """
  Return a dataframe with metrics related to `num_words` number of query
  words
  """

  df: pd.DataFrame = pd.DataFrame(
    columns=[
      "Name",
      "num_words",
      "IoU",
      "WER",
      "Mean Alignment Error",
      "Q1",
      "Median",
      "Q3",
    ],
  )

  for csv_file in os.listdir(J_EVALUATION_DIR):
    # blacklist files starting with "_"
    if csv_file.startswith("_"):
      continue

    csv_path = os.path.join(J_EVALUATION_DIR, csv_file)
    print(f"{csv_path = }")

    data: pd.DataFrame = load_jamendo_dataframe(csv_path)
    if language:
      data = data[data["language"] == language]

    # filter out characters constraints and select num_words rows
    data = data[data["min_char_num"].isna() & data["max_char_num"].isna()]
    data = data[data["num_words"] == num_words]

    # compute wer, iou and average error
    data["IoU"] = data.apply(iou, axis=1)
    data["WER"] = data.apply(
      lambda row: wer(row["model_transcription"], row["lyrics_line"]), axis=1
    )
    data["Mean Alignment Error"] = (
      abs(data["start_time"] - data["model_start_time"])
      + abs(data["end_time"] - data["model_end_time"])
    ) / 2

    avg_error: float = data["Mean Alignment Error"].mean()
    iou_: float = data["IoU"].mean()
    wer_: float = data["WER"].mean()
    median: float = data["Mean Alignment Error"].median()
    q1: float = data["Mean Alignment Error"].quantile(0.25)
    q3: float = data["Mean Alignment Error"].quantile(0.75)

    row: dict[str, Any] = {
      "Name": csv_file,
      "num_words": num_words,
      "IoU": iou_,
      "WER": wer_,
      "Mean Alignment Error": avg_error,
      "Median": median,
      "Q1": q1,
      "Q3": q3,
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
  return df


def main() -> None:
  df: pd.DataFrame = get_models_df(num_words=5)
  df.drop(columns=["num_words"], inplace=True)
  # df.to_html("out.html")
  # print(df)

  # latex_table = df.to_latex(
  #   index=False,
  #   float_format="%.3f",
  #   caption="Model Metrics",
  #   label="tab:model_metrics",
  #   column_format="lrrrrrr",
  #   escape=True,
  # )
  # print(latex_table)


if __name__ == "__main__":
  main()
