import pandas as pd

def clean_dataset(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    original_size = len(df)
    
    # Filter out medically impossible blood pressures
    df_cleaned = df[(df['ap_hi'] < 250) & (df['ap_hi'] > 50)]
    df_cleaned = df_cleaned[(df_cleaned['ap_lo'] < 200) & (df_cleaned['ap_lo'] > 40)]
    
    cleaned_size = len(df_cleaned)
    removed = original_size - cleaned_size
    
    # Save to a new file
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"Cleaning complete!")
    print(f"Original records: {original_size}")
    print(f"Cleaned records:  {cleaned_size}")
    print(f"Removed {removed} outliers.")
    print(f"Saved new dataset to {output_file}")

if __name__ == "__main__":
    clean_dataset('cardio_train.csv', 'cardio_train_cleaned.csv')