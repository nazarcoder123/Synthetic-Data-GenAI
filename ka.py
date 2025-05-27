# # Import necessary libraries
# import os
# import time
# import random
# import pandas as pd
# from pydantic import BaseModel
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from tqdm import tqdm  # For progress tracking
# from dotenv import load_dotenv  # For loading environment variables from .env file
# import logging  # For logging
# import asyncio  # For asynchronous operations
# from google.generativeai import GenerativeModel  # Direct import for async

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("synthetic_data_generation.log"),
#         logging.StreamHandler()
#     ]
# )

# # Load environment variables from .env file
# load_dotenv()
# logging.info(".env file loaded.")

# # Configuration parameters
# TOTAL_RECORDS = 500               # Total number of synthetic records to generate (including outliers)
# BATCH_SIZE = 200                   # Number of records to generate per API call
# MAX_CONCURRENT_REQUESTS = 2       # Parallel requests
# OUTLIER_PERCENTAGE = 0            # Percent of outliers to inject (e.g., 3% = 30 outliers)

# logging.info(f"Configuration: TOTAL_RECORDS={TOTAL_RECORDS}, BATCH_SIZE={BATCH_SIZE}, MAX_CONCURRENT_REQUESTS={MAX_CONCURRENT_REQUESTS}, OUTLIER_PERCENTAGE={OUTLIER_PERCENTAGE}%")

# # Initialize Gemini API
# API_KEY = os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     logging.error("GEMINI_API_KEY not found in environment variables.")
#     exit()
# genai.configure(api_key=API_KEY)
# logging.info("Gemini API configured.")

# # Define the data model
# class MedicalBilling(BaseModel):
#     patient_id: int
#     patient_name: str
#     diagnosis_code: str
#     procedure_code: str
#     total_charge: float
#     insurance_claim_amount: float
#     is_outlier: bool = False  # New field to track outliers

# # Optimized prompt for structured output
# PROMPT_TEMPLATE = """Generate exactly {batch_size} medical billing records.
# Each record must be on a new line.
# Each line must contain ONLY the following 6 pieces of data, separated by a comma and a single space (`, `):
# 1. Patient ID (a 1-5 digit unique number)
# 2. Patient Name (Realistic fictional name)
# 3. Diagnosis Code (Valid ICD-10 code)
# 4. Procedure Code (Valid CPT code)
# 5. Total Charge (a number between 100 and 1000, no dollar sign)
# 6. Insurance Claim Amount (a number that is 60-90% of the Total Charge, no dollar sign)

# Provide the data in this exact CSV format, with NO headers, NO extra text, NO numbering for lines, and NO markdown formatting (like code blocks).

# Example of the REQUIRED format:
# 12345, John Doe, J20.9, 9920, 500.00, 350.00
# 78901, Jane Smith, M54.5, 9921, 150.50, 120.40
# 34567, Bob Lee, E11.9, 99214, 300.75, 250.25

# Generate exactly {batch_size} records now in that specific format:
# """

# # Async batch processor
# async def process_batch(batch_size, semaphore):
#     logging.info(f"Processing batch of size: {batch_size}")
#     async with semaphore:
#         model = GenerativeModel('gemini-2.0-flash-lite')
#         generation_config = {
#             'temperature': 0.7,
#         }
#         response = await model.generate_content_async(
#             PROMPT_TEMPLATE.format(batch_size=batch_size),
#             generation_config=generation_config
#         )
#         return response.text

# # Outlier record generator
# def generate_outlier_records(n):
#     logging.info(f"Generating {n} outlier records.")
#     outliers = []
#     for _ in range(n):
#         try:
#             outlier = MedicalBilling(
#                 patient_id=random.randint(100000, 999999),  # too many digits
#                 patient_name="!!! INVALID NAME !!!",
#                 diagnosis_code="XXX.XX",  # Invalid ICD-10
#                 procedure_code="99999",  # Unusual CPT
#                 total_charge=round(random.uniform(5000, 10000), 2),  # Very high charge
#                 insurance_claim_amount=round(random.uniform(0.01, 0.2) * 10000, 2),  # Unreasonably low
#                 is_outlier=True
#             )
#             outliers.append(outlier)
#         except Exception as e:
#             logging.error(f"Error generating outlier record: {e}")
#     return outliers

# # Main driver
# async def main():
#     logging.info("Starting synthetic data generation.")
#     semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
#     normal_record_count = TOTAL_RECORDS * (100 - OUTLIER_PERCENTAGE) // 100
#     batches_needed = (normal_record_count + BATCH_SIZE - 1) // BATCH_SIZE
#     logging.info(f"Need {normal_record_count} normal records and {TOTAL_RECORDS - normal_record_count} outliers.")

#     tasks = [process_batch(BATCH_SIZE, semaphore) for _ in range(batches_needed)]

#     synthetic_data = []
#     with tqdm(total=normal_record_count, desc="Generating Normal Records") as pbar:
#         for future in asyncio.as_completed(tasks):
#             try:
#                 result = await future
#                 for line in result.strip().split('\n'):
#                     parts = [x.strip() for x in line.split(', ')]
#                     if len(parts) != 6:
#                         logging.warning(f"Skipping malformed line: {line}")
#                         continue
#                     try:
#                         record = MedicalBilling(
#                             patient_id=int(parts[0]),
#                             patient_name=parts[1],
#                             diagnosis_code=parts[2],
#                             procedure_code=parts[3],
#                             total_charge=float(parts[4]),
#                             insurance_claim_amount=float(parts[5]),
#                             is_outlier=False
#                         )
#                         synthetic_data.append(record)
#                     except Exception as e:
#                         logging.error(f"Validation error: {e} in line: {line}")
#                         continue
#                 pbar.update(min(BATCH_SIZE, normal_record_count - pbar.n))
#                 if len(synthetic_data) >= normal_record_count:
#                     break
#             except Exception as e:
#                 logging.error(f"Batch processing failed: {e}")

#     # Generate and add outliers
#     outlier_count = TOTAL_RECORDS - len(synthetic_data)
#     outliers = generate_outlier_records(outlier_count)
#     synthetic_data.extend(outliers)
#     logging.info(f"Total records collected (normal + outliers): {len(synthetic_data)}")

#     # Save to CSV
#     df = pd.DataFrame([r.dict() for r in synthetic_data[:TOTAL_RECORDS]])
#     output_file = "medical_billing_records.csv"
#     try:
#         df.to_csv(output_file, index=False)
#         logging.info(f"Dataset saved to {output_file}")
#     except Exception as e:
#         logging.error(f"Failed to save dataset: {e}")

#     # Display sample
#     logging.info("Sample records:")
#     logging.info(f"\n{df.head(3)}")

# # Entrypoint
# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         logging.warning("Script interrupted by user.")
#     except Exception as e:
#         logging.critical(f"Unhandled error: {e}")
