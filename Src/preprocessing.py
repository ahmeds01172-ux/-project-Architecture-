"""
Refactored Preprocessing Module
Design Patterns Used:
- Strategy Pattern
- Pipeline Pattern
- Factory Pattern
"""

import pandas as pd
import numpy as np
import re
import os
from abc import ABC, abstractmethod

# ==========================================================
# 1. STRATEGY PATTERN
# كل خطوة preprocessing عبارة عن strategy مستقلة
# ==========================================================

class PreprocessingStep(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class PriceCleaning(PreprocessingStep):
    def apply(self, df):
        print("🔧 Cleaning price...")
        df['price'] = df['price'].astype(str).str.replace(',', '').str.strip()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        return df.dropna(subset=['price'])


class SizeExtraction(PreprocessingStep):
    def apply(self, df):
        print("🔧 Extracting size...")

        def extract(size):
            if pd.isna(size):
                return np.nan
            match = re.search(r'/ (\d+(?:\.\d+)?)\s*sqm', str(size))
            return float(match.group(1)) if match else np.nan

        df['size_sqm'] = df['size'].apply(extract)
        return df


class BedroomExtraction(PreprocessingStep):
    def apply(self, df):
        print("🔧 Extracting bedrooms...")

        def extract(b):
            if pd.isna(b):
                return np.nan
            match = re.search(r'(\d+)', str(b))
            return float(match.group(1)) if match else np.nan

        df['bedrooms_num'] = df['bedrooms'].apply(extract)
        return df


class BathroomProcessing(PreprocessingStep):
    def apply(self, df):
        print("🔧 Processing bathrooms...")
        df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
        return df


class DropMissingCritical(PreprocessingStep):
    def apply(self, df):
        print("🔧 Dropping missing critical values...")
        return df.dropna(subset=['price', 'size_sqm'])


class RemoveDuplicates(PreprocessingStep):
    def apply(self, df):
        print("🔧 Removing duplicates...")
        return df.drop_duplicates()


class OutlierRemoval(PreprocessingStep):
    def apply(self, df):
        print("🔧 Removing outliers...")

        def remove(df, col):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return df[(df[col] >= lower) & (df[col] <= upper)]

        df = remove(df, 'price')
        df = remove(df, 'size_sqm')
        return df


class FillMissing(PreprocessingStep):
    def apply(self, df):
        print("🔧 Filling missing values...")

        df['bedrooms_num'] = df['bedrooms_num'].fillna(df['bedrooms_num'].median())
        df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())

        df['type'] = df['type'].fillna(df['type'].mode()[0])
        df['payment_method'] = df['payment_method'].fillna(df['payment_method'].mode()[0])

        return df


class GroupLocations(PreprocessingStep):
    def apply(self, df):
        print("🔧 Grouping rare locations...")
        counts = df['location'].value_counts()
        rare = counts[counts < 30].index
        df['location'] = df['location'].replace(rare, 'Other')
        return df


# ==========================================================
# 2. FACTORY PATTERN
# إنشاء الخطوات في مكان واحد
# ==========================================================

class StepFactory:
    @staticmethod
    def create_pipeline_steps():
        return [
            PriceCleaning(),
            SizeExtraction(),
            BedroomExtraction(),
            BathroomProcessing(),
            DropMissingCritical(),
            RemoveDuplicates(),
            OutlierRemoval(),
            FillMissing(),
            GroupLocations()
        ]


# ==========================================================
# 3. PIPELINE PATTERN
# تنفيذ الخطوات بالتسلسل
# ==========================================================

class PreprocessingPipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self, df):
        for step in self.steps:
            df = step.apply(df)
        return df


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def engineer_features(df):
    print("🔨 Engineering features...")

    def group_locations(loc):
        loc = str(loc).lower()

        if any(x in loc for x in ['new cairo', 'heliopolis', 'maadi']):
            return 'East Cairo'
        elif any(x in loc for x in ['zayed', 'west cairo']):
            return 'West Cairo'
        elif 'october' in loc:
            return '6 October'
        elif 'coast' in loc or 'marina' in loc:
            return 'North Coast'
        else:
            return 'Other'

    df['location_group'] = df['location'].apply(group_locations)

    df['price_per_sqm'] = df['price'] / df['size_sqm']
    df['size_category'] = pd.qcut(df['size_sqm'], 4, labels=['Small', 'Medium', 'Large', 'XL'])
    df['luxury_flag'] = (df['price'] > df['price'].quantile(0.9)).astype(int)

    return df


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def process_data(input_path):
    print("🚀 Starting preprocessing pipeline...\n")

    # Handle relative paths - go up one directory from Src/
    if not os.path.exists(input_path):
        alt_path = os.path.join('..', input_path)
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            # Try from parent directory
            alt_path2 = os.path.join('Data', os.path.basename(input_path))
            if os.path.exists(alt_path2):
                input_path = alt_path2
    
    print(f"📁 Loading data from: {os.path.abspath(input_path)}\n")
    df = pd.read_csv(input_path)

    steps = StepFactory.create_pipeline_steps()
    pipeline = PreprocessingPipeline(steps)

    df = pipeline.run(df)

    df = engineer_features(df)

    os.makedirs('Data', exist_ok=True)
    df.to_csv('Data/processed_Data.csv', index=False)

    print("\n✅ Processing completed successfully!")
    print(f"Final shape: {df.shape}")

    return df


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    # Run from Src directory or parent directory
    input_file = "Data/egypt_real_estate_listings.csv"
    if not os.path.exists(input_file):
        input_file = "../Data/egypt_real_estate_listings.csv"
    
    df = process_data(input_file)
    print(df.head())