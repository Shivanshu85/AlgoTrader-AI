# Learning Exercises - Stock Price Prediction

This directory contains hands-on exercises for the 8-week production ML curriculum.

## Structure

```
exercises/
├── week1/                          # Time Series Foundations
│   ├── detect_autocorrelation.py
│   ├── identify_leakage.py
│   ├── leakage_comparison.py
│   ├── quiz_answers.md
│   └── README.md
├── week2/                          # Proper Validation
│   ├── rolling_window_cv.py
│   ├── backtest_framework.py
│   ├── validation_comparison.py
│   └── README.md
├── week3/                          # Feature Engineering
│   ├── technical_indicators.py
│   ├── statistical_features.py
│   └── feature_pipeline.py
... (weeks 4-8)
└── capstone/                       # Final Project
    ├── data_pipeline.py
    ├── feature_engineering.py
    ├── model_training.py
    ├── model_evaluation.py
    ├── deployment.py
    └── monitoring.py
```

## Getting Started

### Prerequisites

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install project dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import torch, pandas, sklearn; print('✅ All packages installed')"
```

### Running Week 1 Exercises

```bash
# Week 1: Time Series Foundations
cd exercises/week1

# Run autocorrelation detection
python detect_autocorrelation.py

# Run leakage identification
python identify_leakage.py

# Compare leakage impact
python leakage_comparison.py

# Answer the quiz
# Edit quiz_answers.md with your responses
nano quiz_answers.md
```

## Exercise Workflow

For each exercise:

1. **Read the problem** - Understanding what you need to solve
2. **Code the solution** - Write code to solve it
3. **Run and test** - Execute and verify outputs
4. **Analyze results** - Understand what you learned
5. **Answer reflection questions** - Write your understanding

## Success Criteria

Each exercise has specific success criteria. Example:

```python
# Success Criteria Checklist:
success_items = [
    "✅ Generated ACF plots for 3 stocks",
    "✅ Explained autocorrelation findings",
    "✅ Discussed predictability implications",
    "✅ Answered all reflection questions",
]
```

## Getting Help

### If you're stuck:

1. **Re-read the learning material** - often the answer is in the reading
2. **Check the solution sketch** - each file has hints in `TODO` comments
3. **Run with verbose output** - add print statements to understand flow
4. **Check documentation** - most libraries have good docs
5. **Ask on the discussion forum** - peer learning is powerful

### Debugging tips:

```python
import pdb
pdb.set_trace()  # Pause execution to inspect variables

# Or better yet, use pytest with verbose output
pytest exercise_file.py -v -s
```

## Data Files

Stock data files should be placed in:
```
data/
├── AAPL_daily.csv
├── MSFT_daily.csv
├── GOOGL_daily.csv
└── ...
```

## Expected Outputs

Each exercise should produce:

1. **Working code** - All functions implemented and tested
2. **Output files** - Plots, CSVs, or other generated artifacts
3. **Text summary** - Understanding of what you learned
4. **Quiz answers** - Responses to reflection questions

## Time Estimates

- Week 1: 20 hours total (6 hours reading + 14 hours exercises)
- Week 2: 20 hours total (5 hours reading + 15 hours exercises)
- Weeks 3-8: 20-25 hours each
- Capstone: 40 hours

## Verification

To verify you've completed each week:

```bash
# Week 1 check
make verify-week1

# Week 2 check
make verify-week2

# All weeks
make verify-all

# Individual test
pytest exercises/week1/ -v
```

## Next Steps

After completing all 8 weeks:

1. Review your notes and code
2. Identify concepts you want to dive deeper into
3. Start exploring Phase 1: Architecture Design
4. Apply learnings to your own stock prediction project

## Project Structure Context

These exercises build toward the full production system in `/production/`:

```
Stock Prediction System
├── Data Ingestion (Week 2-3)
│   └── production/data_ingestion/
├── Feature Engineering (Week 3)
│   └── production/features/
├── Model Training (Week 4-5)
│   └── production/training/
├── Model Serving (Week 6-7)
│   └── production/serving/
└── Monitoring (Week 8)
    └── production/monitoring/
```

## Additional Resources

- Book: "Forecasting: Principles and Practice" (https://otexts.com/fpp2/)
- Course: Coursera "Sequence Models"
- Blog: Machine Learning Mastery time series tutorials
- Paper: "Attention is All You Need"

## Tips for Success

1. **Write code, don't just read it** - Your fingers learn by typing and debugging
2. **Experiment with parameters** - What happens if you change values?
3. **Visualize everything** - Plots often reveal insights that numbers hide
4. **Keep a learning log** - Document what you learned each day
5. **Teach someone else** - Explain concepts to a colleague or friend
6. **Make mistakes** - Errors are learning opportunities
7. **Don't rush** - Deep understanding takes time

## Checklist: Weekly Completion

Create a file called `PROGRESS.md` to track your progress:

```markdown
# Learning Progress

## Week 1: Time Series Foundations
- [ ] Read all assigned materials
- [ ] Complete autocorrelation exercise
- [ ] Complete leakage identification
- [ ] Understand the implications
- [ ] Answer all quiz questions
- [ ] Could explain to a colleague

## Week 2: Proper Validation
- [ ] Understand temporal validation
- [ ] Implement RollingWindowCV
- [ ] Build backtester
- [ ] Run backtests on multiple stocks
- [ ] Compare different strategies

[... continue for other weeks ...]
```

## Estimated Timeline

```
Week 1:  Data Leakage Prevention          (Feb 15 - Feb 22)
Week 2:  Temporal Validation              (Feb 22 - Mar 1)
Week 3:  Feature Engineering              (Mar 1 - Mar 8)
Week 4:  Advanced LSTM & Attention        (Mar 8 - Mar 15)
Week 5:  Hyperparameter Tuning            (Mar 15 - Mar 22)
Week 6:  Experiment Tracking              (Mar 22 - Mar 29)
Week 7:  Production Deployment            (Mar 29 - Apr 5)
Week 8:  Monitoring & Improvement         (Apr 5 - Apr 12)
Capstone: Complete ML System              (Apr 12 - Apr 26)
```

---

**Ready to start?** Begin with `exercises/week1/README.md`

**Questions?** Check `docs/guides/troubleshooting.md` or file an issue.

Good luck! 🚀
