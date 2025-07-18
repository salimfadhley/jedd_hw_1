import logging
import pathlib
import re
import numpy

import matplotlib.pyplot as plt
import pandas

log = logging.getLogger(__name__)

#The Data.csv file attached the the folder is data from a 14cm hoverhang test. The code is tailored for this overhang in the Young's Modulus calculation.

def normalize_data(data:pandas.DataFrame, sampling_rate:int=5000)->pandas.DataFrame:
    mean:float = numpy.mean(data)
    scaling_factor:int = 3.3/65535
    num_samples:int = len(data)
    time_index = numpy.arange(num_samples) / sampling_rate
    data["voltage"] = (data["sample"] - mean) * scaling_factor

    new_data = data[["voltage"]]
    new_data.index = time_index

    return new_data

def crop_data(normalized_data:pandas.DataFrame)->pandas.DataFrame:
    return normalized_data

def generate_plot_from_data(
    cropped_data: pandas.DataFrame,
    sampling_rate: int = 5000,
):
    """
    Create a scatter-plot of the measured signal.

    Parameters
    ----------
    cropped_data : pandas.DataFrame
        Must contain a column named ``"voltage"`` holding the
        normalised signal in volts.
    sampling_rate : int, optional
        Number of samples per second (Hz).  Default is ``5000``.

    Returns
    -------
    matplotlib.pyplot
        The pyplot module so the caller can perform additional
        adjustments if desired.
    """
    # Build a time axis (in seconds) from the row index
    time_axis = numpy.arange(len(cropped_data)) / sampling_rate
    voltage = cropped_data["voltage"].to_numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(time_axis, voltage, s=6, color="tab:blue", marker="o", alpha=0.9)

    # Aesthetics
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs Time")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4)

    fig.tight_layout()
    plt.show()

    return plt

def calculate_youngs_modulus(cropped_data:pandas.DataFrame)->float:
    return 0.0


def process_file(file_object:pathlib.Path, length:int, experiment_number:int) -> None:
    log.info(f"Processing: {file_object.name=}, {length=}, {experiment_number=}")
    raw_data:pandas.DataFrame = pandas.read_csv(file_object, header=None, names=['sample'])
    log.info(f"Loaded {len(raw_data)} rows of data")
    normalized_data:pandas.DataFrame = normalize_data(raw_data)
    cropped_data:pandas.DataFrame = crop_data(normalized_data)
    plt = generate_plot_from_data(cropped_data)
    # youngs_modules:float = calculate_youngs_modulus(cropped_data)
    #
    # log.info(f"Calculated youngs modulus: {youngs_modules}")


def main():
    logging.basicConfig(level=logging.INFO)



    script_dir:pathlib.Path = pathlib.Path(__file__).parent.resolve()
    set_of_input_files:list[pathlib.Path] = script_dir.glob("*.csv")

    for input_file_object in set_of_input_files:
        filename_parts = re.search(r'(\d{2,})cm(\d)\.csv', input_file_object.name)
        if filename_parts:
            str_length:int = int(filename_parts[1])
            experiment_number:int  = int(filename_parts[2])
            process_file(file_object=input_file_object, length=str_length, experiment_number=experiment_number)
        else:
            log.warn(f"Ignoring file {input_file_object.name}")

if __name__ == "__main__":
    main()