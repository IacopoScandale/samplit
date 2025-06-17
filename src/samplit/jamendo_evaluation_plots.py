import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jiwer import wer

from data.strings import (
  J_DATAFRAME_CSV,
  J_EVALUATION_DIR,
  PLOTS_FOLDER,
  SNS_PALETTE,
  TITLE_FONTSIZE,
)
from data.utils import load_jamendo_dataframe

sns.set_style(style="whitegrid")


def plot_start_end_error_against_query_len(
  out_filename: str | None = None,
  csv_file: str = J_DATAFRAME_CSV,
  language: str | None = None,
  show: bool = False,
) -> None:
  df: pd.DataFrame = load_jamendo_dataframe(csv_file)
  if language:
    df = df[df["language"] == language]

  # df view without character len constraints
  subset = df[df["min_char_num"].isna() & df["max_char_num"].isna()]
  grouped = (
    subset.groupby("num_words")[["start_error", "end_error"]].mean().reset_index()
  )

  # Melt the dataframe to long-form for seaborn
  melted = grouped.melt(
    id_vars="num_words",
    value_vars=["start_error", "end_error"],
    var_name="Error Type",
    value_name="Error (seconds)",
  )

  # Plot start_error and end_error against k
  plt.figure(figsize=(8, 5))
  sns.lineplot(
    data=melted,
    x="num_words",
    y="Error (seconds)",
    hue="Error Type",
    marker="o",
    # linewidth=2,
    palette=SNS_PALETTE,
  )
  plt.title(
    "Start vs End Alignment Error Against Number of Query Words",
    fontsize=TITLE_FONTSIZE,
  )
  plt.xlabel("Number of Query Words")
  plt.ylabel("Average Error (seconds)")
  plt.tight_layout()
  if out_filename:
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    csv_dir = os.path.join(PLOTS_FOLDER, csv_name)
    if not os.path.exists(csv_dir):
      os.mkdir(csv_dir)
    plt.savefig(os.path.join(csv_dir, out_filename))
  if show:
    plt.show()
  else:
    plt.close()


def plot_average_error_boxplots_against_query_len(
  out_filename: str | None = None,
  csv_file: str = J_DATAFRAME_CSV,
  language: str | None = None,
  logy: bool = False,
  show: bool = False,
) -> None:
  df: pd.DataFrame = load_jamendo_dataframe(csv_file)
  if language:
    df = df[df["language"] == language]

  # filter character len constraints
  subset = df[df["min_char_num"].isna() & df["max_char_num"].isna()]

  # number of samples for every boxplot
  counts = subset["num_words"].value_counts().sort_index()

  plt.figure(figsize=(10, 6))
  ax = sns.boxplot(
    data=subset,
    x="num_words",
    y="avg_error",
    hue="num_words",
    palette=SNS_PALETTE,
    legend=False,
  )
  # add sample number n to each boxplot
  xticks = ax.get_xticks()
  y_max = subset["avg_error"].max()
  y_offset = 0.02 * y_max if not logy else y_max * 0.2  # Adjust label height
  for xtick, count in zip(xticks, counts):
    ax.text(
      xtick,
      y_max + y_offset,
      f"n={count}",
      ha="center",
      va="bottom",
      fontsize=9,
      color="black",
    )

  plt.title(
    "Distribution of Alignment Error by Number of Words",
    fontsize=TITLE_FONTSIZE,
  )
  plt.xlabel("Number of Words")
  plt.ylabel("Average Alignment Error (s)")
  if logy:
    plt.yscale("log")
  plt.tight_layout()
  if out_filename:
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    csv_dir = os.path.join(PLOTS_FOLDER, csv_name)
    if not os.path.exists(csv_dir):
      os.mkdir(csv_dir)
    plt.savefig(os.path.join(csv_dir, out_filename))
  if show:
    plt.show()
  else:
    plt.close()


def plot_language_errors(
  out_filename: str | None = None,
  csv_file: str = J_DATAFRAME_CSV,
  show: bool = False,
) -> None:
  df: pd.DataFrame = load_jamendo_dataframe(csv_file)
  # filter character len constraints
  subset = df[df["min_char_num"].isna() & df["max_char_num"].isna()]

  # Language error plot
  plt.figure(figsize=(10, 6))
  sns.lineplot(
    data=subset,
    x="num_words",
    y="avg_error",
    hue="language",
    marker="o",
    errorbar=None,
    linewidth=3,
    palette=SNS_PALETTE,
  )
  plt.title(
    "Alignment Error vs. Number of Words by Language",
    fontsize=TITLE_FONTSIZE,
  )
  plt.tight_layout()
  if out_filename:
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    csv_dir = os.path.join(PLOTS_FOLDER, csv_name)
    if not os.path.exists(csv_dir):
      os.mkdir(csv_dir)
    plt.savefig(os.path.join(csv_dir, out_filename))
  if show:
    plt.show()
  else:
    plt.close()


def plot_query_error_boxplots(
  out_filename: str | None = None,
  csv_file: str = J_DATAFRAME_CSV,
  language: str | None = None,
  show: bool = False,
) -> None:
  df: pd.DataFrame = load_jamendo_dataframe(csv_file)
  if language:
    df = df[df["language"] == language]

  def plot_group(data, title, labels, ax):
    sns.boxplot(
      data=data,
      ax=ax,
      palette=SNS_PALETTE,
    )
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Average Error")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    for i, group in enumerate(data):
      if len(group) == 0:
        continue

      median = np.median(group)
      q3 = np.percentile(group, 75)
      mean = np.mean(group)
      std = np.std(group)

      stats_text = (
        f"median: {median:.2f}\n"
        f"Q3    : {q3:.2f}\n"
        f"mean  : {mean:.2f}\n"
        f"std   : {std:.2f}\n"
        f"({len(group)} samples)"
      )

      y = np.max(group)  # top of the box
      ax.annotate(
        stats_text,
        xy=(i, y),
        xytext=(i + 0.2, y),  # offset to the right
        textcoords="data",
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
      )

  fig, axes = plt.subplots(1, 2, figsize=(14, 6))

  # One-word queries
  group1 = df[(df["num_words"] == 1) & (df["max_char_num"] == 5)]
  group2 = df[
    (df["num_words"] == 1) & (df["min_char_num"] == 5) & (df["max_char_num"] == 10)
  ]

  data1 = [group1["avg_error"].dropna().values, group2["avg_error"].dropna().values]
  labels1 = ["Char number 1-5", "Char number 5-10"]

  plot_group(data1, "One-word Queries", labels1, axes[0])

  # Two-word queries
  group3 = df[(df["num_words"] == 2) & (df["max_char_num"] == 10)]
  group4 = df[(df["num_words"] == 2) & (df["min_char_num"] == 10)]

  data2 = [group3["avg_error"].dropna().values, group4["avg_error"].dropna().values]
  labels2 = ["Char number 1-10", "Char number 10+"]

  plot_group(data2, "Two-word Queries", labels2, axes[1])

  plt.tight_layout()
  if out_filename:
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    csv_dir = os.path.join(PLOTS_FOLDER, csv_name)
    if not os.path.exists(csv_dir):
      os.mkdir(csv_dir)
    plt.savefig(os.path.join(csv_dir, out_filename))
  if show:
    plt.show()
  else:
    plt.close()


def plot_average_wer_boxplots_against_query_len(
  out_filename: str | None = None,
  csv_file: str = J_DATAFRAME_CSV,
  language: str | None = None,
  logy: bool = False,
  show: bool = False,
) -> None:
  df: pd.DataFrame = load_jamendo_dataframe(csv_file)
  if language:
    df = df[df["language"] == language]

  df["word_error_rate"] = df.apply(
    lambda row: wer(row["model_transcription"], row["lyrics_line"]), axis=1
  )

  # filter character len constraints
  subset = df[df["min_char_num"].isna() & df["max_char_num"].isna()]

  # number of samples for every boxplot
  counts = subset["num_words"].value_counts().sort_index()

  plt.figure(figsize=(10, 6))
  ax = sns.boxplot(
    data=subset,
    x="num_words",
    y="word_error_rate",
    hue="num_words",
    palette=SNS_PALETTE,
    legend=False,
  )
  # add sample number n to each boxplot
  xticks = ax.get_xticks()
  y_max = subset["word_error_rate"].max()
  y_offset = 0.02 * y_max if not logy else y_max * 0.2  # Adjust label height
  for xtick, count in zip(xticks, counts):
    ax.text(
      xtick,
      y_max + y_offset,
      f"n={count}",
      ha="center",
      va="bottom",
      fontsize=9,
      color="black",
    )

  plt.title(
    "Distribution of Word Error Rate by Number of Words",
    fontsize=TITLE_FONTSIZE,
  )
  plt.xlabel("Number of Words")
  plt.ylabel("Word Error Rate")
  if logy:
    plt.yscale("log")
  plt.tight_layout()
  if out_filename:
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    csv_dir = os.path.join(PLOTS_FOLDER, csv_name)
    if not os.path.exists(csv_dir):
      os.mkdir(csv_dir)
    plt.savefig(os.path.join(csv_dir, out_filename))
  if show:
    plt.show()
  else:
    plt.close()


def main() -> None:
  for csv_file in os.listdir(J_EVALUATION_DIR):
    csv_path = os.path.join(J_EVALUATION_DIR, csv_file)
    print(f"{csv_path = }")

    # Multilingual: English + French + German + Spanish
    plot_start_end_error_against_query_len("start_end_error_against_query_len", csv_file)
    plot_average_error_boxplots_against_query_len("average_error_boxplots", csv_file)
    plot_average_error_boxplots_against_query_len("average_error_boxplots_log", csv_file, logy=True)
    plot_language_errors("language_errors", csv_file)
    plot_query_error_boxplots("char_constraint_boxplot", csv_file)
    plot_average_wer_boxplots_against_query_len("wer_boxplots", csv_file, logy=False)
    
    # English only
    plot_start_end_error_against_query_len("en_start_end_error_against_query_len", csv_path, language="English")
    plot_average_error_boxplots_against_query_len("en_average_error_boxplots", csv_path, language="English")
    plot_average_error_boxplots_against_query_len("en_average_error_boxplots_log", csv_path, logy=True, language="English")
    plot_query_error_boxplots("en_char_constraint_boxplot", csv_path, language="English")
    plot_average_wer_boxplots_against_query_len("en_wer_boxplots", csv_path, logy=False, language="English")


if __name__ == "__main__":
  main()
