# Financial Analysis Agent

A Python-based financial analysis tool that processes company annual reports and provides comprehensive stock analysis and predictions using Google's Gemini LLM.

## Features

- **PDF Text Extraction**: Automatically extracts and processes text from annual report PDFs
- **Financial Data Analysis**: Calculates key financial ratios and metrics from extracted data
- **Stock Price Analysis**: Performs technical analysis with price predictions
- **AI-Powered Insights**: Uses Google Gemini 1.5 Pro for intelligent financial analysis
- **Investment Recommendations**: Provides actionable buy/hold/sell recommendations
- **Data Visualization**: Creates charts for financial trends and stock price predictions

## Installation

### Prerequisites

- Python 3.8+
- Google API key for Gemini

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/financial-analysis-agent.git
   cd financial-analysis-agent
   ```

2. Install required packages:
   ```
   pip install pandas numpy matplotlib yfinance scikit-learn google-generativeai pypdf2 python-dotenv
   ```

3. Create a `.env` file in the project root directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

Run the script with the path to an annual report PDF and the company's stock ticker:

```
python financial_analysis_agent.py --report_path path/to/annual_report.pdf --ticker AAPL
```

Optional arguments:
- `--output`: Specify the output JSON file path (default: `analysis_report.json`)
- `--skip_visuals`: Skip generating visualization charts (useful for headless environments)

You can also set environment variables:
- `SKIP_VISUALIZATIONS=true`: Alternative way to skip visualization generation
- `GOOGLE_API_KEY`: Your Google API key for Gemini (can also be in .env file)

## Example Output

The script generates:

1. A comprehensive JSON report containing:
   - Financial data and ratios
   - Stock price predictions
   - Strengths and weaknesses analysis
   - Investment recommendations
   - Risk assessment

2. Visualization files:
   - Stock price history and predictions chart
   - Financial ratio trend charts

3. Console summary with key metrics and recommendations

## How It Works

1. **Text Extraction**: The agent extracts text from the annual report PDF
2. **Financial Data Extraction**: Gemini LLM identifies and extracts key financial figures
3. **Stock Data Analysis**: Historical stock data is fetched and analyzed
4. **Ratio Calculation**: Financial ratios are calculated from the extracted data
5. **Prediction Generation**:
   - Technical analysis using regression models
   - Fundamental analysis using Gemini LLM
   - Combined analysis weighing both technical and fundamental factors
6. **Visualization Creation**: Charts are generated to visualize the findings
7. **Report Compilation**: All analysis is combined into a comprehensive report

## Troubleshooting

If you encounter 404 NOT_FOUND errors or other issues:

1. **Stock Data Fetch Errors**: The script now includes robust error handling for stock data retrieval with multiple retry attempts and fallback to dummy data if needed.

2. **PDF Processing Issues**: If the PDF cannot be processed, the script will create dummy financial data to allow analysis to continue.

3. **Visualization Errors**: 
   - Use `--skip_visuals` command line argument to bypass visualization generation
   - Set environment variable `SKIP_VISUALIZATIONS=true`
   - Check for matplotlib backend issues in your environment

4. **API Key Issues**: 
   - Ensure your Google API key is correctly set in the `.env` file or as an environment variable
   - Verify the key has access to Gemini models

5. **Deployment Environment**: 
   - For serverless/cloud deployments, ensure all dependencies are correctly installed
   - Consider setting all file paths as absolute rather than relative

## Requirements

- pandas
- numpy
- matplotlib
- yfinance
- scikit-learn
- google-generativeai
- PyPDF2
- python-dotenv

## License

[MIT License](LICENSE)

## Disclaimer

This tool is for informational purposes only and should not be considered financial advice. Always conduct your own research before making investment decisions.
