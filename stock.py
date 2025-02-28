#!/usr/bin/env python3
"""
Financial Analysis Agent

This script analyzes company annual reports, calculates financial ratios,
and predicts stock trends using Google's Gemini LLM.

Usage:
    python financial_analysis_agent.py --report_path path/to/annual_report.pdf --ticker AAPL
"""

import os
import re
import argparse
import json
import warnings
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import google.generativeai as genai
import PyPDF2
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=GOOGLE_API_KEY)

class FinancialAnalysisAgent:
    """Financial analysis agent that processes annual reports and makes predictions."""
    
    def __init__(self, report_path: str, ticker: str):
        """
        Initialize the financial analysis agent.
        
        Args:
            report_path: Path to the annual report PDF
            ticker: Stock ticker symbol
        """
        self.report_path = report_path
        self.ticker = ticker.upper()
        self.report_text = None
        self.financial_data = None
        self.stock_data = None
        self.ratios = {}
        self.predictions = {}
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete financial analysis pipeline."""
        logger.info(f"Starting analysis for {self.ticker}")
        
        # Extract text from the annual report
        self.extract_text_from_report()
        
        # Extract financial data from the report
        self.extract_financial_data()
        
        # Fetch historical stock data
        self.fetch_stock_data()
        
        # Calculate financial ratios
        self.calculate_financial_ratios()
        
        # Generate predictions using Gemini
        self.generate_predictions()
        
        # Create visualizations
        self.create_visualizations()
        
        # Compile and return the final analysis
        return self.compile_analysis()
    
    def extract_text_from_report(self) -> None:
        """Extract text content from the annual report PDF."""
        logger.info("Extracting text from annual report")
        
        try:
            with open(self.report_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                self.report_text = text
                logger.info(f"Successfully extracted {len(text)} characters from report")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def extract_financial_data(self) -> None:
        """Extract key financial data from the report text using Gemini."""
        logger.info("Extracting financial data from report text")
        
        prompt = f"""
        You are a financial analyst AI. Extract the following financial data from this annual report for {self.ticker}.
        Return the data in JSON format with these fields:
        
        1. revenue: List of annual revenue figures for the past 3 years with corresponding years
        2. net_income: List of annual net income figures for the past 3 years with corresponding years  
        3. total_assets: List of total assets figures for the past 3 years with corresponding years
        4. total_liabilities: List of total liabilities figures for the past 3 years with corresponding years
        5. current_assets: List of current assets figures for the past 3 years with corresponding years
        6. current_liabilities: List of current liabilities figures for the past 3 years with corresponding years
        7. cash_flow_operations: List of cash flow from operations figures for the past 3 years with corresponding years
        8. total_equity: List of shareholder equity figures for the past 3 years with corresponding years
        9. eps: List of earnings per share figures for the past 3 years with corresponding years
        
        For each value, include the year and the numeric value in millions or billions as appropriate.
        Return only the JSON object with no additional text.
        
        Here's the relevant portion of the annual report:
        {self.report_text[:15000]}  # Using first 15000 chars to stay within token limits
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Extract JSON from response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response.text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                self.financial_data = json.loads(json_str)
                logger.info("Successfully extracted financial data")
            else:
                # If no JSON found, try another approach with a more structured prompt
                self._fallback_financial_data_extraction()
        except Exception as e:
            logger.error(f"Error extracting financial data: {e}")
            self._fallback_financial_data_extraction()
    
    def _fallback_financial_data_extraction(self) -> None:
        """Fallback method for financial data extraction with a more structured approach."""
        logger.info("Using fallback method for financial data extraction")
        
        # Create a template for the financial data
        financial_data_template = {
            "revenue": [],
            "net_income": [],
            "total_assets": [],
            "total_liabilities": [],
            "current_assets": [],
            "current_liabilities": [],
            "cash_flow_operations": [],
            "total_equity": [],
            "eps": []
        }
        
        # Try to extract each data point individually
        for key in financial_data_template.keys():
            prompt = f"""
            Extract the {key.replace('_', ' ')} figures for the past 3 years from this annual report text.
            For each year, return the year and the value in millions or billions as appropriate.
            Return only the data in this format: 
            [
                {{"year": "YYYY", "value": X.XX}},
                {{"year": "YYYY", "value": X.XX}},
                {{"year": "YYYY", "value": X.XX}}
            ]
            
            Annual report excerpt:
            {self.report_text[:10000]}
            """
            
            try:
                response = self.gemini_model.generate_content(prompt)
                json_pattern = r'\[.*\]'
                json_match = re.search(json_pattern, response.text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    financial_data_template[key] = json.loads(json_str)
            except Exception as e:
                logger.warning(f"Could not extract {key}: {e}")
                financial_data_template[key] = [
                    {"year": "2023", "value": None},
                    {"year": "2022", "value": None},
                    {"year": "2021", "value": None}
                ]
        
        self.financial_data = financial_data_template
        logger.info("Completed fallback financial data extraction")
    
    def fetch_stock_data(self) -> None:
        """Fetch historical stock data for the ticker."""
        logger.info(f"Fetching historical stock data for {self.ticker}")
        
        try:
            end_date = datetime.now()
            start_date = datetime(end_date.year - 5, end_date.month, end_date.day)
            
            # Fetch data
            self.stock_data = yf.download(self.ticker, start=start_date, end=end_date)
            
            if self.stock_data.empty:
                logger.warning(f"No stock data found for {self.ticker}")
                raise ValueError(f"Could not fetch stock data for {self.ticker}")
                
            logger.info(f"Successfully fetched {len(self.stock_data)} days of stock data")
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            self.stock_data = None
    
    def calculate_financial_ratios(self) -> None:
        """Calculate key financial ratios from the extracted financial data."""
        logger.info("Calculating financial ratios")
        
        if not self.financial_data:
            logger.warning("No financial data available to calculate ratios")
            return
        
        ratios = {}
        
        # Calculate ratios for each year
        years = sorted(set([item["year"] for item in self.financial_data["revenue"]]))
        
        for year in years:
            year_ratios = {}
            
            # Helper function to get value for a specific year and metric
            def get_value(metric, year):
                for item in self.financial_data.get(metric, []):
                    if item.get("year") == year and item.get("value") is not None:
                        return item["value"]
                return None
            
            # Profitability Ratios
            revenue = get_value("revenue", year)
            net_income = get_value("net_income", year)
            total_assets = get_value("total_assets", year)
            total_equity = get_value("total_equity", year)
            
            if revenue and net_income:
                year_ratios["profit_margin"] = (net_income / revenue) * 100
            
            if total_assets and net_income:
                year_ratios["return_on_assets"] = (net_income / total_assets) * 100
                
            if total_equity and net_income:
                year_ratios["return_on_equity"] = (net_income / total_equity) * 100
            
            # Liquidity Ratios
            current_assets = get_value("current_assets", year)
            current_liabilities = get_value("current_liabilities", year)
            
            if current_assets and current_liabilities:
                year_ratios["current_ratio"] = current_assets / current_liabilities
            
            # Debt Ratios
            total_liabilities = get_value("total_liabilities", year)
            
            if total_assets and total_liabilities:
                year_ratios["debt_to_assets"] = (total_liabilities / total_assets) * 100
            
            if total_equity and total_liabilities:
                year_ratios["debt_to_equity"] = (total_liabilities / total_equity) * 100
            
            # Efficiency Ratios
            if total_assets and revenue:
                year_ratios["asset_turnover"] = revenue / total_assets
            
            # Store ratios for this year
            ratios[year] = year_ratios
        
        self.ratios = ratios
        logger.info(f"Successfully calculated ratios for {len(ratios)} years")
    
    def generate_predictions(self) -> None:
        """Generate stock predictions using historical data and Gemini model."""
        logger.info("Generating predictions")
        
        predictions = {}
        
        # 1. Generate technical analysis predictions using stock data
        if self.stock_data is not None and not self.stock_data.empty:
            predictions["technical"] = self._generate_technical_predictions()
        
        # 2. Generate fundamental analysis using financial ratios and Gemini
        if self.financial_data and self.ratios:
            predictions["fundamental"] = self._generate_fundamental_predictions()
        
        # 3. Generate combined prediction with Gemini
        if "technical" in predictions or "fundamental" in predictions:
            predictions["combined"] = self._generate_combined_prediction(predictions)
        
        self.predictions = predictions
        logger.info("Completed prediction generation")
    
    def _generate_technical_predictions(self) -> Dict[str, Any]:
        """Generate technical analysis predictions using stock price data."""
        logger.info("Generating technical predictions")
        
        try:
            # Create a copy of the relevant data
            df = self.stock_data[['Close']].copy()
            df.reset_index(inplace=True)
            
            # Create feature set (using date numerical value and simple moving averages)
            df['Date_num'] = df['Date'].apply(lambda x: x.toordinal())
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Drop rows with NaN values
            df.dropna(inplace=True)
            
            # Prepare data for prediction
            X = df[['Date_num', 'SMA_50', 'SMA_200']].values
            y = df['Close'].values
            
            # Split data: use 80% for training, 20% for testing
            split = int(0.8 * len(df))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Normalize data
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            X_test_scaled = scaler_X.transform(X_test)
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train_scaled)
            
            # Make predictions
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate error
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Prepare future dates for prediction
            last_date = df['Date'].iloc[-1].toordinal()
            future_dates = np.array([last_date + i for i in range(1, 91)])  # Next 90 days
            
            # Get latest values for SMA
            latest_sma_50 = df['SMA_50'].iloc[-1]
            latest_sma_200 = df['SMA_200'].iloc[-1]
            
            # Create future feature set
            future_features = np.array([[date, latest_sma_50, latest_sma_200] for date in future_dates])
            future_features_scaled = scaler_X.transform(future_features)
            
            # Make future predictions
            future_pred_scaled = model.predict(future_features_scaled)
            future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()
            
            # Calculate trend
            last_price = df['Close'].iloc[-1]
            pred_30d = future_pred[29]  # 30 days prediction
            pred_90d = future_pred[-1]   # 90 days prediction
            
            trend_30d_pct = ((pred_30d - last_price) / last_price) * 100
            trend_90d_pct = ((pred_90d - last_price) / last_price) * 100
            
            # Determine trend direction
            if trend_90d_pct > 5:
                trend_direction = "Strong Uptrend"
            elif trend_90d_pct > 0:
                trend_direction = "Slight Uptrend"
            elif trend_90d_pct > -5:
                trend_direction = "Slight Downtrend"
            else:
                trend_direction = "Strong Downtrend"
            
            return {
                "current_price": last_price,
                "prediction_30d": pred_30d,
                "prediction_90d": pred_90d,
                "trend_30d_pct": trend_30d_pct,
                "trend_90d_pct": trend_90d_pct,
                "trend_direction": trend_direction,
                "confidence": 1.0 - (rmse / last_price)  # Normalize RMSE as confidence
            }
        except Exception as e:
            logger.error(f"Error generating technical predictions: {e}")
            return {
                "error": str(e),
                "trend_direction": "Unknown"
            }
    
    def _generate_fundamental_predictions(self) -> Dict[str, Any]:
        """Generate fundamental analysis predictions using Gemini."""
        logger.info("Generating fundamental predictions")
        
        # Format financial data and ratios for Gemini
        financial_summary = json.dumps(self.financial_data, indent=2)
        ratios_summary = json.dumps(self.ratios, indent=2)
        
        prompt = f"""
        You are a financial analyst AI. Analyze the following financial data and ratios for {self.ticker} 
        and provide a fundamental analysis prediction.
        
        Financial Data:
        {financial_summary}
        
        Financial Ratios:
        {ratios_summary}
        
        Based on this data, provide:
        1. An analysis of the company's financial health
        2. Key strengths and weaknesses
        3. Prediction for future performance (growth potential, risks)
        4. Fair value estimate (undervalued, overvalued, or fairly valued)
        5. A trend direction (Strong Uptrend, Slight Uptrend, Slight Downtrend, or Strong Downtrend)
        
        Return your analysis in JSON format with these fields:
        {{
            "financial_health": "your analysis",
            "strengths": ["strength1", "strength2", ...],
            "weaknesses": ["weakness1", "weakness2", ...],
            "growth_potential": "high/medium/low",
            "risks": ["risk1", "risk2", ...],
            "valuation": "undervalued/overvalued/fair",
            "trend_direction": "direction"
        }}
        
        Return only the JSON with no additional explanation.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Extract JSON from response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response.text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("Could not extract JSON from Gemini response for fundamental analysis")
                return {
                    "error": "Could not generate fundamental analysis",
                    "trend_direction": "Unknown"
                }
        except Exception as e:
            logger.error(f"Error generating fundamental predictions: {e}")
            return {
                "error": str(e),
                "trend_direction": "Unknown"
            }
    
    def _generate_combined_prediction(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate combined prediction using both technical and fundamental analysis."""
        logger.info("Generating combined prediction")
        
        technical = predictions.get("technical", {})
        fundamental = predictions.get("fundamental", {})
        
        # Format existing predictions for Gemini
        predictions_summary = json.dumps({
            "technical": technical,
            "fundamental": fundamental
        }, indent=2)
        
        prompt = f"""
        You are a financial analyst AI. Based on both technical and fundamental analysis for {self.ticker},
        provide a combined prediction.
        
        Current Predictions:
        {predictions_summary}
        
        Please create a comprehensive prediction that weighs both the technical and fundamental factors.
        
        Return your analysis in JSON format with these fields:
        {{
            "overall_outlook": "bullish/neutral/bearish",
            "confidence": 0-1 (as a decimal),
            "price_target_6m": numeric value or "unable to determine",
            "price_target_1y": numeric value or "unable to determine",
            "investment_recommendation": "buy/hold/sell",
            "key_factors": ["factor1", "factor2", ...],
            "summary": "brief summary of the combined analysis"
        }}
        
        Return only the JSON with no additional explanation.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Extract JSON from response
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response.text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("Could not extract JSON from Gemini response for combined analysis")
                return {
                    "error": "Could not generate combined analysis",
                    "summary": "Unable to provide a combined prediction."
                }
        except Exception as e:
            logger.error(f"Error generating combined predictions: {e}")
            return {
                "error": str(e),
                "summary": "An error occurred while generating the combined prediction."
            }
    
    def create_visualizations(self) -> None:
        """Create visualizations for the financial analysis."""
        logger.info("Creating visualizations")
        
        try:
            # 1. Stock price history and prediction chart
            if self.stock_data is not None and not self.stock_data.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(self.stock_data.index[-365:], self.stock_data['Close'][-365:], label='Historical Price (1 Year)')
                
                if "technical" in self.predictions:
                    tech_pred = self.predictions["technical"]
                    current_price = tech_pred.get("current_price")
                    
                    if current_price and "prediction_90d" in tech_pred:
                        # Create date range for predictions
                        last_date = self.stock_data.index[-1]
                        future_dates = pd.date_range(start=last_date, periods=91, freq='D')[1:]
                        
                        # Create prediction values (linear interpolation)
                        pred_values = np.linspace(
                            current_price, 
                            tech_pred["prediction_90d"], 
                            num=90
                        )
                        
                        plt.plot(future_dates, pred_values, 'r--', label='Price Prediction (90 Days)')
                
                plt.title(f'{self.ticker} Stock Price History and Prediction')
                plt.xlabel('Date')
                plt.ylabel('Price ($)')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'{self.ticker}_price_prediction.png')
                
                # 2. Financial ratio trends
                if self.ratios:
                    years = sorted(self.ratios.keys())
                    
                    if len(years) > 0:
                        # Profitability ratios
                        plt.figure(figsize=(12, 6))
                        
                        for year in years:
                            profit_margin = self.ratios[year].get("profit_margin")
                            roe = self.ratios[year].get("return_on_equity")
                            roa = self.ratios[year].get("return_on_assets")
                            
                            if profit_margin is not None:
                                plt.bar(year, profit_margin, alpha=0.7, label=f'Profit Margin {year}')
                            
                            # Add ROE and ROA as lines
                            if roe is not None or roa is not None:
                                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                                if roe is not None:
                                    plt.plot(year, roe, 'ro', label=f'ROE {year}')
                                if roa is not None:
                                    plt.plot(year, roa, 'bo', label=f'ROA {year}')
                        
                        plt.title(f'{self.ticker} Profitability Ratios')
                        plt.xlabel('Year')
                        plt.ylabel('Percentage (%)')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(f'{self.ticker}_profitability_ratios.png')
                
                logger.info("Successfully created visualizations")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def compile_analysis(self) -> Dict[str, Any]:
        """Compile all analysis into a comprehensive report."""
        logger.info("Compiling final analysis")
        
        # Get the latest year's financial ratios
        latest_ratios = {}
        if self.ratios:
            latest_year = max(self.ratios.keys()) if self.ratios else None
            if latest_year:
                latest_ratios = self.ratios[latest_year]
        
        # Get prediction summaries
        technical_summary = self.predictions.get("technical", {})
        fundamental_summary = self.predictions.get("fundamental", {})
        combined_summary = self.predictions.get("combined", {})
        
        # Assemble final report
        final_analysis = {
            "company": self.ticker,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "financial_data": self.financial_data,
            "latest_ratios": latest_ratios,
            "stock_analysis": {
                "current_price": technical_summary.get("current_price"),
                "trend_direction": combined_summary.get("overall_outlook") or technical_summary.get("trend_direction"),
                "prediction_30d": technical_summary.get("prediction_30d"),
                "prediction_90d": technical_summary.get("prediction_90d"),
                "investment_recommendation": combined_summary.get("investment_recommendation", "N/A"),
                "confidence": combined_summary.get("confidence", technical_summary.get("confidence", 0))
            },
            "strengths": fundamental_summary.get("strengths", []),
            "weaknesses": fundamental_summary.get("weaknesses", []),
            "risks": fundamental_summary.get("risks", []),
            "summary": combined_summary.get("summary", "No summary available.")
        }
        
        logger.info("Analysis compilation complete")
        return final_analysis

def main():
    """Main function to run the financial analysis agent."""
    parser = argparse.ArgumentParser(description='Financial Analysis Agent')
    parser.add_argument('--report_path', type=str, required=True, help='Path to the annual report PDF')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--output', type=str, default='analysis_report.json', help='Output file path for the analysis')
    args = parser.parse_args()
    
    # Create and run the agent
    agent = FinancialAnalysisAgent(args.report_path, args.ticker)
    
    try:
        analysis = agent.run_analysis()
        
        # Save analysis to file
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*50)
        print(f"FINANCIAL ANALYSIS REPORT FOR {args.ticker}")
        print("="*50)
        print(f"Current Price: ${analysis['stock_analysis']['current_price']:.2f}")
        print(f"Trend Direction: {analysis['stock_analysis']['trend_direction']}")
        print(f"30-Day Price Target: ${analysis['stock_analysis']['prediction_30d']:.2f}")
        print(f"90-Day Price Target: ${analysis['stock_analysis']['prediction_90d']:.2f}")
        print(f"Recommendation: {analysis['stock_analysis']['investment_recommendation']}")
        print(f"Confidence: {analysis['stock_analysis']['confidence']:.2f}")
        print("\nSummary:")
        print(analysis['summary'])
        print("\nAnalysis saved to:", args.output)
        print("Visualizations saved as PNG files.")
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
