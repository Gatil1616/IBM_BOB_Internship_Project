# Student Performance Analytics Dashboard

A professional, portfolio-ready web application that analyzes student academic performance using Python, Pandas, and React. Built as an internship project submission demonstrating core Data Analytics skills.

---

## Project Overview

This dashboard loads, cleans, and analyzes the **Students Performance in Exams** dataset from Kaggle (~1,000 student records). It answers practical analytical questions about student performance patterns across subjects, demographic groups, and preparation habits using descriptive statistics and interactive visualizations.

**Main objective:** Load student data → Clean data → Analyze performance → Create KPIs → Visualize patterns → Generate business/academic insights.

---

## Business / Analytical Questions

The dashboard attempts to answer:

- What is the overall student performance across Math, Reading, and Writing?
- Which subject has the highest average score?
- How many students passed (average score ≥ 60)?
- How does performance differ between male and female students?
- Do students who completed the test preparation course have different average scores?
- How do scores vary by parental education level?
- How does lunch type relate to average scores?
- Which subjects have the strongest correlation with each other?
- Which demographic groups have relatively higher or lower average scores?

> ⚠️ **Important:** This is observational data. Comparisons between groups use language such as "average scores were higher for…" — no causal claims are made.

---

## Dataset

**Source:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

**File:** `StudentsPerformance.csv` (~1,000 records, 8 columns)

| Column | Description |
|--------|-------------|
| `gender` | Student gender (male / female) |
| `race/ethnicity` | Race/ethnicity group (Group A–E) |
| `parental level of education` | Highest parental education level |
| `lunch` | Lunch type (standard / free or reduced) |
| `test preparation course` | Test prep completion (completed / none) |
| `math score` | Math exam score (0–100) |
| `reading score` | Reading exam score (0–100) |
| `writing score` | Writing exam score (0–100) |

---

## Data Cleaning

Performed in [`backend/app.py`](backend/app.py) using Python and Pandas:

1. **Column standardization** — Strip whitespace, lowercase, replace spaces/slashes with underscores
2. **Duplicate removal** — `drop_duplicates()`
3. **Missing value check** — Drop rows where any score column is missing
4. **Data type validation** — Convert score columns to numeric; drop non-numeric rows
5. **Score range validation** — Retain only rows where all scores are 0–100
6. **Categorical standardization** — Strip and normalize case on all categorical columns

### Derived Columns

| Column | Calculation |
|--------|-------------|
| `total_score` | `math_score + reading_score + writing_score` |
| `average_score` | `total_score / 3` (rounded to 2 decimal places) |
| `performance_category` | Based on average score thresholds (see below) |
| `pass_status` | `Pass` if `average_score >= 60`, else `Not Passing` |

### Performance Category Thresholds

| Category | Average Score Range |
|----------|---------------------|
| Excellent | 90 – 100 |
| Good | 75 – 89 |
| Average | 60 – 74 |
| Needs Improvement | 0 – 59 |

> These are analytical categories created for this project. They are not official school grading standards.
> The pass threshold (60) is configurable in `backend/app.py` via `PASS_THRESHOLD`.

---

## Analysis

### Subject Performance
Calculates average Math, Reading, and Writing scores. Identifies the highest and lowest performing subjects.

### Demographic Comparisons
Compares average scores across gender, race/ethnicity groups, parental education levels, and lunch type. Uses neutral, observational language throughout.

### Test Preparation Comparison
Compares students who completed vs did not complete the test preparation course across all three subjects and overall average.

### Performance Categories
Groups students into Excellent, Good, Average, and Needs Improvement based on average score thresholds. Shows count and percentage in each category.

### Subject Correlations
Calculates Pearson correlation coefficients between all subject pairs. Presents scatter plots for visual inspection.

---

## Dashboard Pages

### 1. Dashboard
Main overview page with:
- 7 KPI cards (Total Students, Avg Math, Avg Reading, Avg Writing, Overall Avg, Pass Rate, Highest Subject)
- Average Score by Subject (bar chart)
- Performance Category Distribution (pie chart)
- Average Score by Gender (grouped bar)
- Test Preparation Comparison (grouped bar)
- Parental Education Analysis (horizontal bar)
- Lunch Type Comparison (grouped bar)

### 2. Performance Analytics
- KPI summary cards
- Subject average comparison
- Score distribution histograms (0–10, 10–20, …, 90–100)
- Performance category distribution

### 3. Student Demographics
- Average scores by gender
- Average scores by race/ethnicity group
- Average scores by parental education level
- Average scores by lunch type

### 4. Test Preparation
- Side-by-side statistics for Completed vs Not Completed
- Grouped bar chart comparison
- Dynamic insight text generated from actual calculated difference

### 5. Subject Relationships
- Pearson correlation coefficients for all three subject pairs
- Interactive scatter plots (Math vs Reading, Math vs Writing, Reading vs Writing)

### 6. Key Insights
- Dynamically generated insights from actual dataset calculations
- Covers: best/worst subject, test prep difference, best parental education group, lunch comparison, pass rate, category percentages, strongest correlation

### 7. Student Data
- Full searchable, sortable, filterable dataset table
- Pagination (10/20/50/100 rows per page)
- Filters by gender, race/ethnicity, parental education, lunch, test prep, performance category
- Shows all derived columns: Total Score, Average Score, Performance Category, Pass Status

---

## Key Insights (from full dataset)

*Dynamically generated — these values are calculated from the actual dataset at runtime.*

- **Highest average subject:** Reading (avg ≈ 69.2)
- **Lowest average subject:** Math (avg ≈ 66.1)
- **Test prep difference:** Students who completed test prep had a higher average score (≈ 72.7 vs ≈ 65.0) — a difference of approximately 7.6 points
- **Best parental education:** Master's Degree students had the highest average score (≈ 73.6) in this dataset
- **Lunch type gap:** Standard lunch students had a higher average score (≈ 70.8) than free/reduced lunch students (≈ 62.2) in this dataset
- **Pass rate:** 71.5% of students achieved an average score ≥ 60
- **Strongest correlation:** Reading vs Writing (r ≈ 0.955) — very strong positive association

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend / API | Python 3, Flask |
| Data Analysis | Pandas, NumPy |
| Frontend | React 18 (CDN, no build step) |
| Charting | Recharts 2 |
| Styling | Custom CSS (no framework) |

No database required — data is loaded from CSV into memory at startup.

---

## Screenshots

> Add screenshots after running the application at 1440×900 resolution.

### Dashboard
<!-- Screenshot: dashboard.png -->
![Dashboard](screenshots/dashboard.png)

### Performance Analytics
<!-- Screenshot: performance.png -->
![Performance Analytics](screenshots/performance.png)

### Student Demographics
<!-- Screenshot: demographics.png -->
![Student Demographics](screenshots/demographics.png)

### Test Preparation
<!-- Screenshot: test-prep.png -->
![Test Preparation](screenshots/test-prep.png)

### Subject Relationships
<!-- Screenshot: relationships.png -->
![Subject Relationships](screenshots/relationships.png)

### Key Insights
<!-- Screenshot: insights.png -->
![Key Insights](screenshots/insights.png)

### Student Data Explorer
<!-- Screenshot: data.png -->
![Student Data](screenshots/data.png)

---

## How to Run

### Prerequisites
- Python 3.8 or higher
- `pip`

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/student-performance-analytics.git
cd student-performance-analytics

# 2. Install Python dependencies
pip install flask flask-cors pandas numpy

# 3. Start the server
python backend/app.py
```

### Open the dashboard

Navigate to **http://localhost:5000** in your browser.

The Flask server serves both the API and the React frontend. No separate frontend build step is required.

---

## Project Structure

```
student-performance-analytics/
├── StudentsPerformance.csv     # Dataset (from Kaggle)
├── backend/
│   ├── app.py                  # Flask API + data cleaning + serving frontend
│   ├── requirements.txt        # Python dependencies
│   └── test_data.py            # Data validation script
├── frontend/
│   ├── index.html              # HTML entry point
│   ├── App.jsx                 # React application (all pages)
│   └── styles.css              # Application styles
├── screenshots/                # Add screenshots here for README
└── README.md
```

---

## Development Note

This project was developed with assistance from **IBM Bob**, an AI-assisted development tool. The project requirements, analytical objectives, dataset selection, validation, testing, and final refinements were reviewed during development.

---

## Limitations

- This is **observational data**. All comparisons between groups are descriptive only.
- The dataset represents exam performance in a specific sample and may not generalise to all student populations.
- Group comparisons (by gender, race/ethnicity, etc.) should **not** be interpreted as causal relationships.
- Performance categories (Excellent, Good, Average, Needs Improvement) and the pass threshold (average ≥ 60) are **analytical definitions created for this project** — not official grading standards.
- The dataset does not include attendance, study hours, or historical performance data.

---

## Future Improvements

*These are potential future enhancements — **none are currently implemented**.*

- Student performance prediction using regression or classification models
- Attendance data integration
- Study-hours tracking and analysis
- Historical performance tracking across terms
- More advanced statistical tests (e.g., hypothesis testing between groups)
- Export functionality (CSV/PDF reports)

---

## License

This project is intended for educational and portfolio purposes.
Dataset: [Kaggle — Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
