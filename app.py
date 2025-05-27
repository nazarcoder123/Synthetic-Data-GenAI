# Import necessary libraries
import os
import time
import pandas as pd
from pydantic import BaseModel
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm  # For progress tracking
from dotenv import load_dotenv  # For loading environment variables from .env file
import logging  # For logging
import asyncio # For asynchronous operations
from google.generativeai import GenerativeModel # Direct import for async

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("synthetic_data_generation.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()
logging.info(".env file loaded.")

# Configuration parameters
# API key is loaded from the .env file using the key 'GEMINI_API_KEY'
TOTAL_RECORDS = 1000    # Total number of synthetic records to generate
BATCH_SIZE = 200        # Number of records to generate per API call (test with API limits)
MAX_CONCURRENT_REQUESTS = 2  # Parallel requests

logging.info(f"Configuration: TOTAL_RECORDS={TOTAL_RECORDS}, BATCH_SIZE={BATCH_SIZE}, MAX_CONCURRENT_REQUESTS={MAX_CONCURRENT_REQUESTS}")

# Initialize Gemini API with the API key from environment variables
# Note: This uses the LangChain wrapper for Gemini
# If you want to use only the API key, consider using google-generativeai directly
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    exit()
genai.configure(api_key=API_KEY)
logging.info("Gemini API configured.")

# Define the data model for a medical billing record
class MedicalBilling(BaseModel):
    patient_id: int
    patient_name: str
    diagnosis_code: str
    procedure_code: str
    total_charge: float
    insurance_claim_amount: float

# Optimized prompt for structured output - Made more explicit for strict formatting
PROMPT_TEMPLATE = """Generate exactly {batch_size} medical billing records.
Each record must be on a new line.
Each line must contain ONLY the following 6 pieces of data, separated by a comma and a single space (`, `):
1. Patient ID (a 1-5 digit unique number)
2. Patient Name (Realistic fictional name)
3. Diagnosis Code (Valid ICD-10 code)
4. Procedure Code (Valid CPT code)
5. Total Charge (a number between 100 and 1000, no dollar sign)
6. Insurance Claim Amount (a number that is 60-90% of the Total Charge, no dollar sign)

Provide the data in this exact CSV format, with NO headers, NO extra text, NO numbering for lines, and NO markdown formatting (like code blocks).

Example of the REQUIRED format:
12345, John Doe, J20.9, 9920, 500.00, 350.00
78901, Jane Smith, M54.5, 9921, 150.50, 120.40
34567, Bob Lee, E11.9, 99214, 300.75, 250.25

Generate exactly {batch_size} records now in that specific format:
"""

async def process_batch(batch_size, semaphore):
    logging.info(f"Processing batch of size: {batch_size}")
    async with semaphore:
        logging.debug("Acquired semaphore for batch processing.")
        model = GenerativeModel('gemini-2.0-flash-lite')
        generation_config = {
            'temperature': 0.7,
        }
        logging.debug(f"Sending request to Gemini API with temperature: {generation_config['temperature']}")
        response = await model.generate_content_async(
            PROMPT_TEMPLATE.format(batch_size=batch_size),
            generation_config=generation_config
        )
        logging.debug("Received response from Gemini API.")
        return response.text

async def main():
    logging.info("Starting synthetic data generation.")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    batches_needed = (TOTAL_RECORDS + BATCH_SIZE - 1) // BATCH_SIZE
    logging.info(f"Calculated {batches_needed} batches needed.")
    tasks = [process_batch(BATCH_SIZE, semaphore) for _ in range(batches_needed)]
    logging.info(f"Created {len(tasks)} async tasks for batch processing.")
    
    synthetic_data = []
    with tqdm(total=TOTAL_RECORDS, desc="Generating Records") as pbar:
        logging.info("Starting batch processing with progress bar.")
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                logging.debug(f"Processing batch result: {result[:100]}...") # Log first 100 chars
                for line in result.strip().split('\n'):
                    parts = [x.strip() for x in line.split(', ')]
                    if len(parts) != 6:
                        logging.warning(f"Skipping line due to incorrect format: {line}")
                        continue
                    
                    # Corrected instantiation of MedicalBilling
                    try:
                        record = MedicalBilling(
                            patient_id=int(parts[0]),
                            patient_name=parts[1],
                            diagnosis_code=parts[2],
                            procedure_code=parts[3],
                            total_charge=float(parts[4]),
                            insurance_claim_amount=float(parts[5])
                        )
                        synthetic_data.append(record)
                        logging.debug(f"Successfully parsed and validated record: {record.dict()}")
                    except Exception as e:
                        # Log parsing errors if any
                        logging.error(f"Error validating parsed entry: {line}. Error: {e}")
                        continue # Skip this record if validation fails
                    
                pbar.update(min(BATCH_SIZE, TOTAL_RECORDS - pbar.n))
                logging.debug(f"Progress bar updated. Current total records: {len(synthetic_data)}")
                if len(synthetic_data) >= TOTAL_RECORDS:
                    logging.info(f"Target of {TOTAL_RECORDS} records reached. Stopping early.")
                    break
                      
            except Exception as e:
                logging.error(f"Batch error: {str(e)}")
      
      # Create DataFrame and save
    logging.info(f"Finished batch processing. Total records collected: {len(synthetic_data)}")
    df = pd.DataFrame([r.dict() for r in synthetic_data[:TOTAL_RECORDS]])
    logging.info(f"Created pandas DataFrame with {len(df)} records.")
    
    output_file = "medical_billing_records.csv"
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Dataset saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving file {output_file}: {str(e)}")

    # Display a sample of the generated records
    logging.info("Sample records:")
    logging.info(f"\n{df.head(3)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}") 
