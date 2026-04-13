# Capstone-2

**Project Update (Weeks 1–2): Data Preparation Completed***

**Integrated two datasets:**

  * World CO₂ Emissions dataset (base dataset from last semester)
  * Renewable Energy Consumption dataset (1990–2022)
* Cleaned and standardized both datasets (country identifiers and year formatting).
* Converted renewable dataset to a tidy/long format with:

  * Year as a single column
  * Renewable_Consumption_Pct as the value column
* Merged datasets using *Country Code + Year* to create a combined country-year dataset.
* To improve reliability, we applied data-quality filtering:

  * Focused analysis range: *1990–2020*
  * Excluded a small set of territories where multiple CO₂ indicators showed identical values across years (likely reporting/coverage issues).
    
*** Final output file:**
  * final_merged_co2_renewable_1990_2020_filtered.csv

***Next Steps (Week 3 onward)***

* Perform EDA (trends, distributions, missingness).
* Correlation analysis between renewable energy consumption and CO₂ metrics.
* Build baseline ML models and evaluate performance.
* Summarize insights and provide recommendations.
