import os
import numpy as np
import pandas as pd

def generate_student_concept_matrix(folder_path, output_csv_path):
    student_dict = {}
    concept_dict = {}

    # Step 1: First pass to collect all unique student_ids and concept_ids
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Traverse each block of 3 lines
                for i in range(0, len(lines), 3):
                    # Ensure there are enough lines to process a complete block
                    """
                    if i + 1 >= len(lines):
                        continue
                    """
                    student_id = lines[i].strip().split(',')[0]  # Get the studentid from the first line
                    concept_ids = lines[i + 1].strip().split(',')  # Get multiple concept_ids from the third line

                    # Add student_id and concept_ids to the dictionaries
                    if student_id not in student_dict:
                        student_dict[student_id] = len(student_dict)
                    for concept_id in concept_ids:
                        if concept_id not in concept_dict:
                            concept_dict[concept_id] = len(concept_dict)

    # Step 2: Initialize the interaction matrix with zeros
    num_students = len(student_dict)
    num_concepts = len(concept_dict)
    interaction_matrix = np.zeros((num_students, num_concepts), dtype=int)

    # Step 3: Second pass to fill in the interaction counts
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Traverse each block of 4 lines
                for i in range(0, len(lines), 3):
                    # Ensure there are enough lines to process a complete block
                    """
                    if i + 1 >= len(lines):
                        continue
                    """

                    student_id = lines[i].strip().split(',')[0]  # Get the studentid from the first line
                    concept_ids = lines[i + 1].strip().split(',')  # Get multiple concept_ids from the third line

                    # Get the corresponding student index
                    student_index = student_dict[student_id]

                    # Step 4: Increase the interaction count for each concept associated with this student
                    for concept_id in concept_ids:
                        if concept_id in concept_dict:
                            concept_index = concept_dict[concept_id]
                            interaction_matrix[student_index][concept_index] += 1

    # Step 4: Sort student ids and reorder the interaction matrix accordingly
    sorted_student_ids = sorted(student_dict.keys(), key=lambda x: int(x))  # Sort student_ids as integers
    sorted_interaction_matrix = np.zeros((len(sorted_student_ids), num_concepts), dtype=int)

    # Reorder the interaction matrix rows based on the sorted student ids
    for i, student_id in enumerate(sorted_student_ids):
        original_index = student_dict[student_id]
        sorted_interaction_matrix[i, :] = interaction_matrix[original_index, :]

    # Step 5: Sort concept ids and reorder the columns in the interaction matrix
    sorted_concept_ids = sorted(concept_dict.keys(), key=lambda x: int(x))  # Sort concept_ids as integers
    concept_index_map = [concept_dict[concept_id] for concept_id in sorted_concept_ids]

    # Reorder the columns based on sorted concept ids
    sorted_interaction_matrix = sorted_interaction_matrix[:, concept_index_map]

    # Step 6: Convert the sorted numpy array into a DataFrame
    df = pd.DataFrame(sorted_interaction_matrix, columns=sorted_concept_ids)

    # Step 7: Save the DataFrame to a CSV file without the student_id index column
    df.to_csv(output_csv_path, index=False,header=False)

    # Step 8: Print DataFrame shape and head to verify the result
    print("Generated matrix shape:", df.shape)
    print("Preview of the generated matrix (first 10 rows and columns):")
    print(df.iloc[:10, :10])  # Display first 10 rows and columns for verification

    return df

# Example usage:
folder_path = "./"  # Replace with the actual path to your csv files
output_csv_path = "student_concept_matrix.csv"  # Specify the path to save the csv
matrix_df = generate_student_concept_matrix(folder_path, output_csv_path)

# Optionally, you can load the saved CSV to verify correctness
loaded_df = pd.read_csv(output_csv_path, sep='\t')
print("Loaded CSV matrix shape:", loaded_df.shape)
print("Loaded CSV preview (first 10 rows and columns):")
print(loaded_df.iloc[:10, :10])
