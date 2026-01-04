# Dataset Download Instructions

## IMDb Movie Review Dataset

This project uses the IMDb movie review dataset from Google Sheets.

## Dataset Source

**Google Sheets Link:** [IMDb Dataset](https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/edit?gid=1889101679#gid=1889101679)

## Download Methods

### Method 1: Manual Download (Recommended)

1. Open the Google Sheets link in your browser
2. Go to **File → Download → Comma Separated Values (.csv)**
3. Save the file as `imdb_data.csv` in the `data/` directory
4. Make sure the file has the following structure:
   - Column 1: `review` - The movie review text
   - Column 2: `sentiment` - The sentiment label (positive/negative)

### Method 2: Direct Export URL

If the Google Sheet is publicly accessible, you can download it directly using:

```bash
curl -L "https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/export?format=csv&gid=1889101679" -o data/imdb_data.csv
```

Or use Python:

```python
import pandas as pd

url = "https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/export?format=csv&gid=1889101679"
df = pd.read_csv(url)
df.to_csv('data/imdb_data.csv', index=False)
```

### Method 3: Using the Download Script

Run the provided download script:

```bash
python download_dataset.py
```

## Dataset Structure

The dataset should contain:
- **review**: Text of the movie review (may contain HTML tags)
- **sentiment**: Sentiment label - either "positive" or "negative"

## Expected File Location

```
imdb_sentiment_analysis/
└── data/
    └── imdb_data.csv
```

## Verification

After downloading, verify the dataset:

```python
import pandas as pd

df = pd.read_csv('data/imdb_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
```

## Notes

- The dataset may contain HTML tags in the review text (these will be cleaned during preprocessing)
- Make sure the sentiment column contains only "positive" or "negative" values
- The dataset should be balanced or at least have a reasonable distribution of both classes

