# Soccer Performance Analysis Dashboard

## Overview
This is a comprehensive Streamlit application designed to analyze the performance of soccer clubs in the Brazilian Série A. The application offers detailed insights into team performance across multiple dimensions, including defense, defensive transition, offensive transition, attack, and chance creation.

## Key Features

### Club Analysis Module
Allows users to analyze an individual club's performance in four different ways:
- **Club vs Club**: Compare a specific match against other matches the club has played in the competition.
- **Club in Round**: Compare a specific match against other matches in the same round.
- **Club in Competition**: Compare the club's performance against other clubs in the competition using a 5-match moving average.
- **2025 vs 2024**: Compare the club's current season performance with the previous season.

### Opponent Analysis Module
Provides deep analysis of potential opponents:
- Analyze opponents' performances in home or away matches
- View their strongest and weakest metrics
- Generate AI-powered opponent analysis reports
- Download analysis reports in PDF or TXT format

## Technical Information

### Dependencies
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- soccerplots
- SciPy
- Google's Generative AI
- FPDF

### Data Structure
The application processes three main types of data:
1. Team performance metrics (z-scored for normalization)
2. Match-specific data
3. Contextual information for AI analysis

### AI Integration
The application integrates with Google's Gemini API to generate detailed opponent analyses based on statistical metrics and tactical context.

## Usage

1. Select either "Club Analysis" or "Opponent Analysis"
2. Choose a club from the dropdown menu
3. Follow the specific prompts for your selected analysis type:
   - For Club Analysis: Select the comparison mode and specific match if applicable
   - For Opponent Analysis: Specify whether to analyze home or away performance

## Visualization

The application provides extensive data visualizations:
- Interactive scatter plots comparing metrics across teams
- Detailed metrics breakdowns for specific performance areas
- Historical performance trends
- Color-coded visual indicators to highlight strengths and weaknesses

## Output Options

- Interactive web dashboard
- Downloadable PDF reports
- Text-based analytical summaries

## Notes
- The data represents performance metrics for the Brazilian Série A 2025 season
- All metrics are normalized as z-scores (standard deviations from the mean)
- The moving average calculations use the 5 most recent matches
