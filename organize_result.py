import os
import json
import pandas as pd

# Define root directory containing all experiments
root_dir = "results"  # Change this to your actual root directory

# Define experiment names
experiments = [
    "tum_test_6", "tum_test_6_dpdm_top_63", "tum_test_6_dpdm_top_127",
    "tum_test_6_dpdm_top_255", "tum_test_6_dpdm_top_512", "tum_test_6_dpdm_top_all"
]

# Define sequence names
sequences = ["fr1_desk", "fr1_desk2", "fr1_room", "fr2_desk", "fr2_xyz", "fr3_long_office_household"]

# Define result types
result_types = ["rpe", "ape"]  # We store both RPE and APE in separate cells

# Define experiment modes
modes = ["7dof"]

for mode in modes:
    # Create a DataFrame with experiments as rows and sequences as columns
    multi_index = pd.MultiIndex.from_product([experiments, result_types], names=["Experiment", "Metric"])
    df = pd.DataFrame(index=multi_index, columns=sequences)

    # Iterate through experiments, sequences, and result types
    for experiment in experiments:
        for sequence in sequences:
            row_rpe, row_ape = None, None  # Default values for missing data

            for result_type in result_types:
                stats_path = os.path.join(root_dir, experiment, sequence, mode, result_type, "stats.json")

                if os.path.exists(stats_path):
                    with open(stats_path, "r") as f:
                        stats = json.load(f)

                    # Extract mean value and assign correctly
                    mean_value = stats.get("mean", None)
                    if mean_value is not None:
                        mean_value = f"{mean_value:.4f}"  # Format to 4 decimal places

                    if result_type == "rpe":
                        row_rpe = mean_value
                    elif result_type == "ape":
                        row_ape = mean_value

            # Store RPE and APE values in the DataFrame
            df.loc[(experiment, "rpe"), sequence] = row_rpe
            df.loc[(experiment, "ape"), sequence] = row_ape

    # Save results to an Excel file using the mode name
    output_file = f"experiment_results_{mode}.xlsx"
    df.to_excel(output_file)
    print(f"Saved results to {output_file}")
