# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import re
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import sqlite3
from datetime import datetime

# -----------------------------------------------------
# DATABASE & ENHANCED FEATURES
# -----------------------------------------------------
class FinancialDB:
    def __init__(self):
        self.conn = sqlite3.connect('finance.db', check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS snapshots 
            (id INTEGER PRIMARY KEY, date TEXT, income REAL, savings REAL, debt REAL, score REAL)''')
    
    def save(self, income, savings, debt, score):
        self.conn.execute("INSERT INTO snapshots VALUES (NULL,?,?,?,?,?)", 
                         (datetime.now().isoformat(), income, savings, debt, score))
        self.conn.commit()
    
    def get_history(self):
        return pd.read_sql("SELECT * FROM snapshots ORDER BY date DESC LIMIT 10", self.conn)

db = FinancialDB()

class DebtCalc:
    @staticmethod
    def snowball(debts, payment):
        """Pay smallest debt first"""
        debts = sorted(debts, key=lambda x: x['balance'])
        months, interest = 0, 0
        while any(d['balance'] > 0 for d in debts):
            months += 1
            for d in debts:
                if d['balance'] > 0:
                    interest += d['balance'] * d['rate']/1200
                    d['balance'] -= payment
            if months > 500: break
        return {'months': months, 'interest': round(interest, 2)}

class SmartCat:
    CATS = {
        "Food": ["food", "restaurant", "طعام", "مطعم", "سوبرماركت"], 
        "Transport": ["car", "gas", "سيارة", "بنزين", "مواصلات"],
        "Bills": ["electric", "water", "كهرباء", "ماء", "إنترنت"],
        "Housing": ["rent", "mortgage", "إيجار", "سكن", "بيت"],
        "Entertainment": ["entertainment", "movies", "ترفيه", "سينما", "تسلية"]
    }
    
    @staticmethod
    def categorize(text):
        for cat, keywords in SmartCat.CATS.items():
            if any(k in text.lower() for k in keywords):
                return cat
        return "Other"

class PeerCompare:
    BENCHMARKS = {
        "low": {"save_rate": 10, "debt_ratio": 35},
        "mid": {"save_rate": 20, "debt_ratio": 25}, 
        "high": {"save_rate": 25, "debt_ratio": 20}
    }
    
    @staticmethod
    def compare(income, monthly_save, debt):
        bracket = "low" if income < 5000 else ("mid" if income < 10000 else "high")
        benchmark = PeerCompare.BENCHMARKS[bracket]
        user_save_rate = (monthly_save/income)*100
        user_debt_ratio = (debt/income)*100
        return {
            'bracket': bracket,
            'save_status': 'above' if user_save_rate >= benchmark['save_rate'] else 'below',
            'debt_status': 'better' if user_debt_ratio <= benchmark['debt_ratio'] else 'worse',
            'user_save_rate': round(user_save_rate, 1),
            'benchmark_save_rate': benchmark['save_rate']
        }

class Alerts:
    @staticmethod
    def generate(income, savings, debt, monthly_save):
        alerts = []
        emergency_months = savings / (income * 0.6) if income > 0 else 0
        
        if emergency_months < 3:
            alerts.append({'icon': '🚨', 'title': 'Emergency Fund Low', 
                          'msg': f'Only {emergency_months:.1f} months saved. Build 3-6 months.'})
        
        debt_ratio = (debt/income)*100 if income > 0 else 0
        if debt_ratio > 40:
            alerts.append({'icon': '💳', 'title': 'High Debt', 
                          'msg': f'Debt is {debt_ratio:.0f}% of income. Reduce to <36%.'})
        
        save_rate = (monthly_save/income)*100 if income > 0 else 0
        if save_rate >= 20:
            alerts.append({'icon': '🎉', 'title': 'Great Savings!', 
                          'msg': f'{save_rate:.0f}% savings rate is excellent!'})
        elif save_rate < 10:
            alerts.append({'icon': '⚠️', 'title': 'Low Savings', 
                          'msg': f'{save_rate:.0f}% savings rate is low. Aim for 15-20%.'})
        
        return alerts

class FinCalc:
    @staticmethod
    def emergency_fund(monthly_expenses):
        return {
            '3_months': monthly_expenses * 3,
            '6_months': monthly_expenses * 6,
            '12_months': monthly_expenses * 12
        }
    
    @staticmethod
    def compound_interest(principal, monthly, rate, years):
        r = rate/100/12
        n = years * 12
        fv = principal * (1+r)**n + monthly * (((1+r)**n - 1)/r)
        return round(fv, 2)

# -----------------------------------------------------
# BLACK DARK PROFESSIONAL THEME
# -----------------------------------------------------
def inject_dark_theme():
    st.markdown("""
    <style>
    /* Black Dark Theme Variables */
    :root {
        --primary: #3b82f6;
        --primary-dark: #2563eb;
        --primary-light: #60a5fa;
        --secondary: #64748b;
        --accent: #10b981;
        --accent-dark: #059669;
        --warning: #f59e0b;
        --danger: #ef4444;
        --background: #0a0a0a;
        --card-bg: #1a1a1a;
        --card-hover: #252525;
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --border: #374151;
        --border-light: #4b5563;
        --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.5), 0 2px 4px -2px rgb(0 0 0 / 0.5);
        --gradient: linear-gradient(135deg, #3b82f6, #10b981);
    }

    /* Main Container */
    .main {
        background-color: var(--background);
        color: var(--text-primary);
    }

    /* Headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1.2rem 0;
        border-left: 4px solid var(--primary);
        padding-left: 1.2rem;
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), transparent);
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
    }

    /* Cards */
    .card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.2rem 0;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }

    .card:hover {
        background: var(--card-hover);
        transform: translateY(-4px);
        box-shadow: 0 12px 25px -8px rgba(0, 0, 0, 0.6);
        border-color: var(--primary);
    }

    .card:hover::before {
        transform: scaleX(1);
    }

    /* Metrics and KPIs */
    .metric-card {
        background: linear-gradient(135deg, var(--card-bg), var(--card-hover));
        color: var(--text-primary);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin: 0.8rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px -6px rgba(59, 130, 246, 0.3);
        border-color: var(--primary-light);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.8rem 0;
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* Interactive Buttons */
    .stButton button {
        background: var(--gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        position: relative;
        overflow: hidden;
    }

    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -8px var(--primary);
    }

    .stButton button:hover::before {
        left: 100%;
    }

    .stButton button:active {
        transform: translateY(0);
    }

    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: var(--card-bg);
        border-right: 1px solid var(--border);
    }

    /* Input Fields */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox select {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        background: var(--card-hover) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-bottom: 2px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        border: 2px solid transparent !important;
        transition: all 0.3s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient) !important;
        color: white !important;
        border-color: var(--primary) !important;
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--card-hover) !important;
        color: var(--text-primary) !important;
    }

    /* Alerts and Notifications */
    .health-alert {
        padding: 1.8rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border-left: 6px solid;
        background: var(--card-bg);
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }

    .health-alert:hover {
        transform: translateX(4px);
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.4);
    }

    /* Progress Bars */
    .progress-container {
        background: var(--border);
        border-radius: 10px;
        height: 10px;
        margin: 1.5rem 0;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background: var(--gradient);
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
    }

    /* Loading Spinner */
    .stSpinner > div {
        border: 3px solid var(--border);
        border-top: 3px solid var(--primary);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--card-bg);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-light);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderContent {
        background: var(--card-hover) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] [data-baseweb="track"] {
        background: var(--border) !important;
        height: 6px !important;
    }

    .stSlider [data-baseweb="slider"] [data-baseweb="thumb"] {
        background: var(--primary) !important;
        border: 3px solid white !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }

    /* Success/Error Messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        color: var(--accent) !important;
        border: 1px solid var(--accent) !important;
        border-radius: 12px !important;
    }

    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        color: var(--danger) !important;
        border: 1px solid var(--danger) !important;
        border-radius: 12px !important;
    }

    /* Info Cards */
    .info-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.2rem 0;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }

    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -8px rgba(0, 0, 0, 0.4);
        border-color: var(--primary-light);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        
        .card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            padding: 1.5rem 1rem;
            margin: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# ENHANCED UI COMPONENTS FOR DARK THEME
# -----------------------------------------------------
def create_metric_card(value: str, label: str, change: str = None):
    """Create a beautiful metric card"""
    change_html = ""
    if change:
        change_color = "color: #10b981;" if change.startswith("+") else "color: #ef4444;"
        change_html = f'<div style="{change_color} font-size: 0.9rem; margin-top: 0.5rem; font-weight: 600;">{change}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

def create_info_card(title: str, content: str):
    """Create an information card"""
    st.markdown(f"""
    <div class="info-card">
        <div style="flex: 1;">
            <h4 style="margin: 0 0 1rem 0; color: var(--text-primary); font-size: 1.3rem; font-weight: 700;">{title}</h4>
            <p style="margin: 0; color: var(--text-secondary); line-height: 1.6; font-size: 1rem;">{content}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# 1) Hugging Face API with Smart Model Selection
# -----------------------------------------------------
MODELS = {
    "mixtral": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama2-70b": "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf",
    "mistral": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    "zephyr": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
}

class SmartModelSelector:
    def __init__(self):
        self.language_models = {
            "arabic": ["mixtral", "mistral"],
            "english": ["llama2-70b", "mixtral", "zephyr"]
        }

        self.topic_models = {
            "personal_budget": ["mistral", "zephyr"],
            "investment": ["llama2-70b", "mixtral"],
            "retirement": ["llama2-70b", "mixtral"]
        }

    def detect_language(self, text: str) -> str:
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        if arabic_pattern.search(text):
            return "arabic"
        return "english"

    def detect_topic(self, goal: str) -> str:
        goal_lower = goal.lower()
        if any(word in goal_lower for word in ["استثمار", "investment", "سهم", "stock", "portfolio", "invest"]):
            return "investment"
        elif any(word in goal_lower for word in ["تقاعد", "retirement", "شيخوخة", "pension"]):
            return "retirement"
        else:
            return "personal_budget"

    def select_best_model(self, user_input: dict) -> list:
        language = self.detect_language(
            user_input.get("goal", "") +
            user_input.get("fixed_expenses", "") +
            user_input.get("variable_expenses", "")
        )

        topic = self.detect_topic(user_input.get("goal", ""))

        lang_models = self.language_models.get(language, [])
        topic_models = self.topic_models.get(topic, [])

        candidate_models = list(set(lang_models) & set(topic_models))
        if not candidate_models:
            candidate_models = lang_models or ["mixtral"]

        return candidate_models

class EnhancedHFInference:
    def __init__(self):
        self.model_selector = SmartModelSelector()
        self.models = MODELS
        self.session = requests.Session()
        self.timeout = 45

    def call_model(self, prompt: str, model_url: str) -> Optional[str]:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.3,
                    "top_p": 0.85,
                    "do_sample": True,
                    "return_full_text": False
                }
            }

            headers = {"Content-Type": "application/json"}

            response = self.session.post(
                model_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return str(result)
            elif response.status_code == 503:
                return None
            else:
                return None

        except Exception as e:
            return None

    def smart_inference(self, prompt: str, user_input: dict) -> str:
        model_priority = self.model_selector.select_best_model(user_input)

        for model_name in model_priority:
            model_url = self.models[model_name]
            result = self.call_model(prompt, model_url)
            if result and len(result.strip()) > 10:
                return result

        return ""

# -----------------------------------------------------
# 2) Pydantic Schema
# -----------------------------------------------------
class SpendingChange(BaseModel):
    category: str = Field(..., description="Expense category name")
    reduce_by: str = Field(..., description="Reduction percentage")
    current_monthly: Optional[float] = Field(None, description="Current monthly spending")
    new_monthly: float = Field(..., description="New monthly spending after reduction")
    annual_savings: Optional[float] = Field(None, description="Annual savings from this change")

class SavingPlan(BaseModel):
    monthly_save: float = Field(..., description="Monthly savings amount")
    months_to_goal: int = Field(..., description="Months to reach financial goal")
    total_goal: Optional[float] = Field(None, description="Total goal amount")

class InvestmentPlan(BaseModel):
    low_risk: int = Field(..., ge=0, le=100, description="Low risk investment percentage")
    medium_risk: int = Field(..., ge=0, le=100, description="Medium risk investment percentage")
    high_risk: int = Field(..., ge=0, le=100, description="High risk investment percentage")
    expected_return: Optional[str] = Field(None, description="Expected annual return range")

class FinancialHealth(BaseModel):
    savings_rate_percentage: float = Field(..., description="Percentage of income saved")
    emergency_fund_status: str = Field(..., description="Status of emergency fund")
    debt_to_income_ratio: str = Field(..., description="Debt to income ratio assessment")

class AdvisorOutput(BaseModel):
    saving_plan: SavingPlan
    spending_changes: List[SpendingChange]
    investment_plan: InvestmentPlan
    action_steps: List[str]
    financial_health: Optional[FinancialHealth] = Field(None, description="Overall financial health assessment")

parser = PydanticOutputParser(pydantic_object=AdvisorOutput)

# -----------------------------------------------------
# 3) Prompt Template
# -----------------------------------------------------
template = """
You are an expert certified financial planner (CFP) with 20+ years experience.
Analyze the user's financial situation and provide SPECIFIC, ACTIONABLE advice.

**USER FINANCIAL PROFILE:**
- Monthly Income: ${income}
- Fixed Expenses Breakdown: {fixed_expenses}
- Variable Expenses Breakdown: {variable_expenses}
- Current Savings: ${savings}
- Total Debt: ${debt}
- Financial Goal: {goal}
- Risk Tolerance: {risk}

**YOUR TASK:**
1. Create a REALISTIC monthly savings plan (15-30% of income typically)
2. Analyze the specific expense categories provided and identify 3-4 categories to optimize with EXACT percentages
3. Provide investment allocation matching their risk profile
4. Give 5-7 actionable, time-bound steps
5. Assess overall financial health

**CRITICAL REQUIREMENTS:**
- Base spending recommendations on ACTUAL expense categories provided by user
- Savings must be sustainable (not overly aggressive)
- Provide SPECIFIC numbers, not ranges
- Investment allocation MUST sum to exactly 100%
- Include current vs new spending amounts for EACH recommended category
- Be encouraging but realistic

**OUTPUT FORMAT:**
{format_instructions}

Respond ONLY with valid JSON. Do not include any other text.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["income", "fixed_expenses", "variable_expenses", "savings", "debt", "goal", "risk"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# -----------------------------------------------------
# 4) Sensitivity Analysis & Health Monitoring
# -----------------------------------------------------
class SensitivityAnalyzer:
    @staticmethod
    def calculate_sensitivity(base_plan: dict, savings_ratio_change: float) -> dict:
        income = base_plan.get('income', 5000)
        current_savings = base_plan['saving_plan']['monthly_save']
        current_ratio = current_savings / income

        new_ratio = max(0.05, min(0.5, current_ratio + savings_ratio_change))
        new_savings = income * new_ratio

        goal_amount = base_plan['saving_plan'].get('total_goal', current_savings * 12)
        new_months = max(6, int(goal_amount / new_savings)) if new_savings > 0 else 999

        impact_analysis = {
            'current_savings_ratio': round(current_ratio * 100, 1),
            'new_savings_ratio': round(new_ratio * 100, 1),
            'current_monthly_save': current_savings,
            'new_monthly_save': round(new_savings, 0),
            'current_months': base_plan['saving_plan']['months_to_goal'],
            'new_months': new_months,
            'time_impact': base_plan['saving_plan']['months_to_goal'] - new_months,
            'annual_impact': round((new_savings - current_savings) * 12, 0)
        }

        return impact_analysis

    @staticmethod
    def create_sensitivity_chart(sensitivity_data: dict) -> go.Figure:
        ratios = [sensitivity_data['current_savings_ratio'], sensitivity_data['new_savings_ratio']]
        months = [sensitivity_data['current_months'], sensitivity_data['new_months']]
        labels = ['Current Plan', 'Adjusted Plan']

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=labels,
            y=ratios,
            name='Savings Rate (%)',
            marker_color=['#3b82f6', '#10b981'],
            text=[f'{r}%' for r in ratios],
            textposition='auto',
        ))

        fig.add_trace(go.Scatter(
            x=labels,
            y=months,
            mode='lines+markers+text',
            name='Months to Goal',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=12),
            text=[f'{m} months' for m in months],
            textposition='top center',
            yaxis='y2'
        ))

        fig.update_layout(
            title='Sensitivity Analysis: Savings Rate Impact',
            xaxis_title='Scenario',
            yaxis_title='Savings Rate (%)',
            yaxis2=dict(
                title='Months to Goal',
                overlaying='y',
                side='right',
                range=[0, max(months) * 1.2]
            ),
            template='plotly_dark',
            height=400,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

class FinancialHealthMonitor:
    @staticmethod
    def calculate_health_rating(user_input: dict, savings_plan: dict) -> dict:
        income = user_input.get('income', 5000)
        savings = user_input.get('savings', 0)
        debt = user_input.get('debt', 0)
        monthly_savings = savings_plan['monthly_save']

        savings_rate = (monthly_savings / income) * 100 if income > 0 else 0
        debt_to_income = (debt / income) * 100 if income > 0 else 0
        emergency_fund_months = savings / (income * 0.6) if income > 0 else 0

        savings_score = min(100, savings_rate * 5)
        debt_score = max(0, 100 - (debt_to_income * 0.5))
        emergency_score = min(100, emergency_fund_months * 25)

        overall_score = (savings_score + debt_score + emergency_score) / 3

        if overall_score >= 80:
            rating = "Excellent"
            color = "#10b981"
        elif overall_score >= 60:
            rating = "Good"
            color = "#3b82f6"
        elif overall_score >= 40:
            rating = "Fair"
            color = "#f59e0b"
        else:
            rating = "Needs Improvement"
            color = "#ef4444"

        warnings = []
        if savings_rate < 10:
            warnings.append("Savings rate is low. Aim for at least 15-20% of income.")
        if debt_to_income > 40:
            warnings.append("Debt-to-income ratio is high. Focus on debt reduction.")
        if emergency_fund_months < 3:
            warnings.append("Emergency fund is insufficient. Build 3-6 months of expenses.")
        if monthly_savings < 100:
            warnings.append("Savings amount is very low. Look for expense reduction opportunities.")

        return {
            'overall_score': round(overall_score, 1),
            'rating': rating,
            'color': color,
            'component_scores': {
                'savings_rate': round(savings_score, 1),
                'debt_management': round(debt_score, 1),
                'emergency_fund': round(emergency_score, 1)
            },
            'warnings': warnings,
            'metrics': {
                'savings_rate': round(savings_rate, 1),
                'debt_to_income': round(debt_to_income, 1),
                'emergency_months': round(emergency_fund_months, 1)
            }
        }

    @staticmethod
    def create_health_dashboard(health_data: dict) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = health_data['overall_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Financial Health Score", 'font': {'color': 'white', 'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                'bar': {'color': health_data['color']},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.3)"},
                    {'range': [40, 60], 'color': "rgba(245, 158, 11, 0.3)"},
                    {'range': [60, 80], 'color': "rgba(59, 130, 246, 0.3)"},
                    {'range': [80, 100], 'color': "rgba(16, 185, 129, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': health_data['overall_score']
                }
            }
        ))

        fig.update_layout(
            height=300, 
            margin=dict(t=80, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        return fig

    @staticmethod
    def create_component_chart(health_data: dict) -> go.Figure:
        components = list(health_data['component_scores'].keys())
        scores = list(health_data['component_scores'].values())

        readable_components = ['Savings Rate', 'Debt Management', 'Emergency Fund']

        fig = go.Figure(data=[
            go.Bar(
                x=readable_components,
                y=scores,
                marker_color=['#10b981', '#3b82f6', '#f59e0b'],
                text=[f'{score}/100' for score in scores],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title='Health Component Scores',
            xaxis_title='Components',
            yaxis_title='Score (0-100)',
            template='plotly_dark',
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

# -----------------------------------------------------
# 5) Financial Advisor Chain
# -----------------------------------------------------
class FinancialAdvisorChain:
    def __init__(self):
        self.cache = {}
        self.parser = parser
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.health_monitor = FinancialHealthMonitor()
        self.hf_client = EnhancedHFInference()

    def validate_user_input(self, user_input: dict) -> dict:
        try:
            # Parse expenses more accurately
            fixed_total, fixed_details = self._parse_expenses_detailed(user_input["fixed_expenses"])
            variable_total, variable_details = self._parse_expenses_detailed(user_input["variable_expenses"])

            income_val = float(user_input["income"])

            # Create detailed expense strings for the prompt
            fixed_str = ", ".join([f"{cat}:${amt}" for cat, amt in fixed_details.items()])
            variable_str = ", ".join([f"{cat}:${amt}" for cat, amt in variable_details.items()])

            return {
                "income": max(100, income_val),
                "fixed_expenses": fixed_str,
                "variable_expenses": variable_str,
                "savings": max(0, float(user_input["savings"])),
                "debt": max(0, float(user_input["debt"])),
                "goal": user_input["goal"],
                "risk": user_input["risk"].split(' - ')[-1].lower().replace(" ", "_"),
                "total_expenses": fixed_total + variable_total,
                "expense_details": {
                    "fixed": fixed_details,
                    "variable": variable_details
                }
            }
        except Exception as e:
            raise ValueError(f"Invalid input data: {str(e)}")

    def _parse_expenses_detailed(self, expense_str: str) -> tuple:
        """Parse expense string and return (total, {category: amount})"""
        if not expense_str:
            return 0, {}

        total = 0
        details = {}
        
        # Handle different separators
        entries = re.split(r',|\n', expense_str)
        
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
                
            # Match patterns like "Rent:2500", "Rent: 2500", "Rent 2500", "Rent - 2500"
            match = re.search(r'([^:\d\n]+)[:\-\s]*(\d+\.?\d*)', entry)
            if match:
                category = match.group(1).strip()
                amount = float(match.group(2))
                details[category] = amount
                total += amount

        return total, details

    def process(self, user_input: dict) -> AdvisorOutput:
        try:
            validated_input = self.validate_user_input(user_input)

            # Create cache key that includes expense details
            cache_key = f"{validated_input['income']}_{validated_input['fixed_expenses']}_{validated_input['variable_expenses']}_{validated_input['savings']}_{validated_input['debt']}_{validated_input['risk']}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            formatted_prompt = prompt.format(**validated_input)
            model_response = self.hf_client.smart_inference(formatted_prompt, validated_input)

            if not model_response:
                raise Exception("LLM inference failed")

            result = self._parse_response(model_response, validated_input)
            self.cache[cache_key] = result
            return result

        except Exception as e:
            return self._create_fallback_response(user_input)

    def _parse_response(self, response: str, user_input: dict) -> AdvisorOutput:
        try:
            return self.parser.parse(response)
        except:
            json_text = self._extract_json(response)
            if json_text:
                try:
                    return self.parser.parse(json_text)
                except:
                    pass
            raise Exception("JSON parsing failed")

    def _extract_json(self, text: str) -> Optional[str]:
        json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\[.*\]|\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return match
            except:
                continue
        return None

    def analyze_sensitivity(self, base_result: dict, savings_ratio_change: float, user_input: dict) -> dict:
        base_data = {
            'income': user_input.get('income', 5000),
            'saving_plan': base_result['saving_plan']
        }

        sensitivity = self.sensitivity_analyzer.calculate_sensitivity(base_data, savings_ratio_change)
        sensitivity_chart = self.sensitivity_analyzer.create_sensitivity_chart(sensitivity)

        return {
            'sensitivity_analysis': sensitivity,
            'sensitivity_chart': sensitivity_chart
        }

    def _create_fallback_response(self, user_input: dict) -> AdvisorOutput:
        try:
            validated_input = self.validate_user_input(user_input)
        except:
            validated_input = {
                "income": 5000.0, "total_expenses": 3000.0, "risk": "balanced",
                "fixed_expenses": "N/A", "variable_expenses": "Food:800, Entertainment:200",
            }

        income = validated_input['income']
        risk = validated_input['risk']
        total_expenses = validated_input['total_expenses']

        available_to_save = income - total_expenses

        if available_to_save >= income * 0.20:
            savings_rate = 0.20
        elif available_to_save >= income * 0.15:
            savings_rate = 0.15
        elif available_to_save > 0:
            savings_rate = available_to_save / income
        else:
            savings_rate = 0.15

        monthly_save = round(income * savings_rate, 0)

        risk_profiles = {
            "conservative": (60, 30, 10),
            "balanced": (40, 40, 20),
            "aggressive": (20, 40, 40)
        }
        low, medium, high = risk_profiles.get(risk, (40, 40, 20))

        # Use actual expense categories from user input
        expense_details = validated_input.get('expense_details', {})
        variable_categories = expense_details.get('variable', {})
        
        spending_changes = []
        if variable_categories:
            for category, amount in list(variable_categories.items())[:3]:  # Use first 3 categories
                reduction_pct = 15 if amount > 300 else 10
                new_amount = round(amount * (1 - reduction_pct/100), 0)
                spending_changes.append(
                    SpendingChange(
                        category=category,
                        reduce_by=f"{reduction_pct}%",
                        current_monthly=amount,
                        new_monthly=new_amount,
                        annual_savings=(amount - new_amount) * 12
                    )
                )
        else:
            # Fallback categories
            spending_changes = [
                SpendingChange(
                    category="Food",
                    reduce_by="15%",
                    current_monthly=500,
                    new_monthly=425,
                    annual_savings=900
                ),
                SpendingChange(
                    category="Entertainment",
                    reduce_by="25%",
                    current_monthly=300,
                    new_monthly=225,
                    annual_savings=900
                )
            ]

        return AdvisorOutput(
            saving_plan=SavingPlan(
                monthly_save=monthly_save,
                months_to_goal=12 if monthly_save > 0 else 24,
                total_goal=round(monthly_save * 12, 0)
            ),
            spending_changes=spending_changes,
            investment_plan=InvestmentPlan(
                low_risk=low,
                medium_risk=medium,
                high_risk=high,
                expected_return="5-7% annually"
            ),
            financial_health=FinancialHealth(
                savings_rate_percentage=round(savings_rate * 100, 1),
                emergency_fund_status="Building phase",
                debt_to_income_ratio="Good"
            ),
            action_steps=[
                f"Automate ${monthly_save:.0f} monthly savings on payday.",
                "Review all monthly subscriptions and cancel 1-2 unused ones.",
                "Cook at home more often to reduce food expenses.",
                "Build emergency fund covering 3-6 months of expenses.",
                "Start with low-cost index funds to match your goal.",
                "Review budget progress weekly and adjust spending.",
            ]
        )

advisor_chain = FinancialAdvisorChain()

# -----------------------------------------------------
# 6) Visualization for Dark Theme
# -----------------------------------------------------
class AdvancedFinancialVisualizer:
    @staticmethod
    def create_savings_progress(saving_plan: dict) -> go.Figure:
        months = list(range(1, saving_plan['months_to_goal'] + 1))
        savings = [saving_plan['monthly_save'] * month for month in months]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=savings,
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='#10b981', width=4),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))

        goal_amount = saving_plan.get('total_goal', savings[-1])
        fig.add_hline(y=goal_amount, line_dash="dash", line_color="#ef4444",
                     annotation_text="Goal", annotation_position="bottom right")

        fig.update_layout(
            title='Savings Progress Over Time',
            xaxis_title='Months',
            yaxis_title='Total Savings ($)',
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_investment_allocation(investment_plan: dict) -> go.Figure:
        labels = ['Low Risk', 'Medium Risk', 'High Risk']
        values = [investment_plan['low_risk'], investment_plan['medium_risk'], investment_plan['high_risk']]
        colors = ['#3b82f6', '#10b981', '#ef4444']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            insidetextorientation='radial'
        )])

        fig.update_layout(
            title='Investment Allocation Strategy',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_spending_optimization(spending_changes: list) -> go.Figure:
        categories = [item['category'] for item in spending_changes]
        current = [item.get('current_monthly', 0) for item in spending_changes]
        new = [item['new_monthly'] for item in spending_changes]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Current Spending',
            x=categories,
            y=current,
            marker_color='#ef4444',
            text=[f"${c:,.0f}" for c in current],
            textposition='auto',
        ))
        fig.add_trace(go.Bar(
            name='Recommended Spending',
            x=categories,
            y=new,
            marker_color='#10b981',
            text=[f"${n:,.0f}" for n in new],
            textposition='auto',
        ))

        fig.update_layout(
            title='Spending Optimization Opportunities',
            xaxis_title='Categories',
            yaxis_title='Monthly Amount ($)',
            barmode='group',
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_income_breakdown(income: float, monthly_save: float, fixed_expenses: float, variable_expenses: float) -> go.Figure:
        disposable_income = income - monthly_save - fixed_expenses - variable_expenses
        disposable_income = max(0, disposable_income)

        labels = ['Savings', 'Fixed Expenses', 'Variable Expenses', 'Disposable Income']
        values = [monthly_save, fixed_expenses, variable_expenses, disposable_income]
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors,
            textinfo='label+value'
        )])

        fig.update_layout(
            title='Monthly Income Allocation',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

visualizer = AdvancedFinancialVisualizer()

# -----------------------------------------------------
# 7) Enhanced Streamlit App with Black Dark Theme
# -----------------------------------------------------
def create_health_recommendations(health_data: dict) -> str:
    recommendations = f"""
    ## Financial Health Recommendations

    ### Overall Rating: {health_data['rating']}
    **Score: {health_data['overall_score']}/100**

    ### Key Metrics:
    - Savings Rate: {health_data['metrics']['savings_rate']}% (target: 15-20%)
    - Debt-to-Income: {health_data['metrics']['debt_to_income']}% (target: <36%)
    - Emergency Fund: {health_data['metrics']['emergency_months']} months (target: 3-6 months)

    ### Priority Actions:
    """

    for warning in health_data['warnings']:
        recommendations += f"- {warning}\n"

    if health_data['overall_score'] >= 80:
        recommendations += "\n**Excellent!** Maintain your current financial habits."
    elif health_data['overall_score'] >= 60:
        recommendations += "\n**Good progress!** Focus on addressing the warnings above."
    else:
        recommendations += "\n**Needs attention!** Prioritize building emergency savings and reducing debt."

    return recommendations

def run_analysis(income, fixed, variable, savings, debt, goal, risk):
    user_input_dict = {
        "income": income,
        "fixed_expenses": fixed,
        "variable_expenses": variable,
        "savings": savings,
        "debt": debt,
        "goal": goal,
        "risk": risk
    }

    result = advisor_chain.process(user_input_dict)
    analysis_data = result.model_dump()

    health_monitor = FinancialHealthMonitor()
    health_data = health_monitor.calculate_health_rating(user_input_dict, analysis_data['saving_plan'])
    analysis_data['health_monitor'] = health_data

    # 🆕 NEW: Save to database
    db.save(income, savings, debt, health_data['overall_score'])
    
    # 🆕 NEW: Add alerts
    analysis_data['alerts'] = Alerts.generate(income, savings, debt, analysis_data['saving_plan']['monthly_save'])
    
    # 🆕 NEW: Add peer comparison
    analysis_data['peer_compare'] = PeerCompare.compare(income, analysis_data['saving_plan']['monthly_save'], debt)

    return analysis_data

def format_advice_output(result):
    if "error" in result:
        return f"Error: {result['error']}"

    output = "## Personalized Financial Plan - خطة مالية مخصصة\n\n"

    if 'financial_health' in result:
        health = result['financial_health']
        output += f"### Financial Health - الصحة المالية العامة\n"
        output += f"- **Savings Rate - معدل الادخار**: {health['savings_rate_percentage']:.1f}%\n"
        output += f"- **Emergency Fund - صندوق الطوارئ**: {health['emergency_fund_status']}\n"
        output += f"- **Debt-to-Income - نسبة الدين للدخل**: {health['debt_to_income_ratio']}\n\n"

    output += "### Savings Plan - خطة الادخار\n"
    savings_plan = result["saving_plan"]
    output += f"- **Monthly Savings - الادخار الشهري**: ${savings_plan['monthly_save']:,.0f}\n"
    output += f"- **Total Goal - الهدف الكلي**: ${savings_plan.get('total_goal', savings_plan['monthly_save'] * savings_plan['months_to_goal']):,.0f}\n"
    output += f"- **Timeline - المدة**: {savings_plan['months_to_goal']} months\n\n"

    output += "### Spending Optimization - تحسين المصروفات\n"
    total_annual_savings = 0
    for change in result["spending_changes"]:
        output += f"- **{change['category']}**: Reduce by {change['reduce_by']}\n"
        if change.get('current_monthly') is not None:
            output += f"  - Current: ${change['current_monthly']:,.0f} → New: ${change['new_monthly']:,.0f}\n"
        if change.get('annual_savings') is not None:
            total_annual_savings += change['annual_savings']
            output += f"  - Annual Savings: ${change['annual_savings']:,.0f}\n"

    if total_annual_savings > 0:
        output += f"\n**Total Annual Savings - إجمالي التوفير السنوي**: ${total_annual_savings:,.0f}\n\n"

    output += "### Investment Plan - خطة الاستثمار\n"
    invest = result["investment_plan"]
    output += f"- **Low Risk - مخاطرة منخفضة**: {invest['low_risk']}%\n"
    output += f"- **Medium Risk - مخاطرة متوسطة**: {invest['medium_risk']}%\n"
    output += f"- **High Risk - مخاطرة عالية**: {invest['high_risk']}%\n"
    if invest.get('expected_return'):
        output += f"- **Expected Return - العائد المتوقع**: {invest['expected_return']}\n"
    output += "\n"

    output += "### Action Steps - خطوات تنفيذية\n"
    for i, step in enumerate(result["action_steps"], 1):
        output += f"{i}. {step}\n"

    return output

def create_dashboard_overview(result, user_input):
    """Create a beautiful dashboard overview with key metrics"""
    savings_plan = result["saving_plan"]
    health_data = result.get('health_monitor', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            value=f"${savings_plan['monthly_save']:,.0f}",
            label="Monthly Savings",
            change=f"+{savings_plan['monthly_save']/user_input['income']*100:.1f}% of income"
        )
    
    with col2:
        create_metric_card(
            value=f"{savings_plan['months_to_goal']}",
            label="Months to Goal",
            change=f"Target: ${savings_plan.get('total_goal', 0):,.0f}"
        )
    
    with col3:
        total_savings = sum([change.get('annual_savings', 0) for change in result["spending_changes"]])
        create_metric_card(
            value=f"${total_savings:,.0f}",
            label="Annual Savings",
            change="From optimization"
        )
    
    with col4:
        score = health_data.get('overall_score', 0)
        create_metric_card(
            value=f"{score:.0f}/100",
            label="Health Score",
            change=health_data.get('rating', 'Good')
        )

def main():
    st.set_page_config(
        page_title="AI Financial Advisor",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject dark theme CSS
    inject_dark_theme()

    st.markdown('<h1 class="main-header">AI Financial Advisor - مستشار مالي ذكي</h1>', unsafe_allow_html=True)
    st.markdown("### Enter your financial data and get a comprehensive financial plan with advanced analytics")

    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = {}
    if 'current_expenses' not in st.session_state:
        st.session_state.current_expenses = {"fixed": "", "variable": ""}
    if 'debts' not in st.session_state:  # 🆕 NEW
        st.session_state.debts = []

    with st.sidebar:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Financial Data Input")
        
        income = st.number_input(
            "Monthly Income - الدخل الشهري",
            min_value=500,
            max_value=100000,
            value=6000,
            step=500
        )

        fixed = st.text_area(
            "Fixed Expenses - المصاريف الثابتة",
            value="Rent:2500, Bills:400, Loans:500",
            placeholder="Category:Amount, Category:Amount..."
        )

        variable = st.text_area(
            "Variable Expenses - المصاريف المتغيرة",
            value="Food:1000, Entertainment:400, Shopping:300",
            placeholder="Category:Amount, Category:Amount..."
        )

        savings = st.number_input(
            "Current Savings - المدخرات الحالية",
            min_value=0,
            value=8000,
            step=1000
        )

        debt = st.number_input(
            "Total Debt - الديون الإجمالية",
            min_value=0,
            value=10000,
            step=1000
        )

        goal = st.text_input(
            "Financial Goal - الهدف المالي",
            value="Buy a car worth $30000 in 2 years",
            placeholder="Example: Save $50000 in 2 years"
        )

        risk = st.selectbox(
            "Risk Tolerance - مستوى المخاطرة",
            options=["Conservative - محافظ", "Balanced - متوازن", "Aggressive - مجازف"],
            index=1
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Analysis Tools")
        
        savings_adjustment = st.slider(
            "Adjust Savings Rate (-10% to +10%)",
            min_value=-0.10,
            max_value=0.10,
            value=0.0,
            step=0.01
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Check if inputs have changed to trigger re-analysis
        current_inputs = {
            "income": income, "fixed": fixed, "variable": variable,
            "savings": savings, "debt": debt, "goal": goal, "risk": risk
        }
        
        # Auto-detect changes in expenses and trigger re-analysis
        expenses_changed = (fixed != st.session_state.current_expenses["fixed"] or 
                          variable != st.session_state.current_expenses["variable"])
        
        if st.button("Analyze Financial Status - تحليل الوضع المالي", type="primary", use_container_width=True) or expenses_changed:
            with st.spinner("Analyzing your financial situation..."):
                try:
                    st.session_state.user_input = current_inputs
                    st.session_state.current_expenses = {"fixed": fixed, "variable": variable}
                    st.session_state.analysis_result = run_analysis(income, fixed, variable, savings, debt, goal, risk)
                    st.success("Analysis completed successfully!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result

        # Dashboard Overview
        create_dashboard_overview(result, st.session_state.user_input)

        health_data = result.get('health_monitor', {})
        if health_data:
            alert_color = health_data.get('color', '#3b82f6')
            st.markdown(
                f"""
                <div class="health-alert" style="border-left-color: {alert_color}">
                    <h3 style="color: {alert_color}; margin: 0; font-size: 1.4rem;">Financial Health: {health_data['rating']}</h3>
                    <p style="margin: 8px 0; font-size: 1.1rem;">Overall Score: <strong>{health_data['overall_score']}/100</strong></p>
                    {''.join([f'<p style="margin: 4px 0; font-size: 1rem;">{warning}</p>' for warning in health_data['warnings']])}
                </div>
                """,
                unsafe_allow_html=True
            )

        # 🆕 UPDATED TABS WITH NEW FEATURES
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Full Report", 
            "📈 Visual Analytics", 
            "🎯 Sensitivity",
            "🆕 Smart Features",  # NEW TAB
            "🆕 Calculators"       # NEW TAB
        ])

        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(format_advice_output(result))
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(visualizer.create_savings_progress(result["saving_plan"]), use_container_width=True)
                st.plotly_chart(visualizer.create_spending_optimization(result["spending_changes"]), use_container_width=True)
            with col2:
                st.plotly_chart(visualizer.create_investment_allocation(result["investment_plan"]), use_container_width=True)

                income_val = st.session_state.user_input.get('income', 5000)
                fixed_expenses_total = sum([float(x) for x in re.findall(r':\s*(\d+\.?\d*)', st.session_state.user_input.get('fixed', ''))])
                variable_expenses_total = sum([float(x) for x in re.findall(r':\s*(\d+\.?\d*)', st.session_state.user_input.get('variable', ''))])

                st.plotly_chart(visualizer.create_income_breakdown(
                    income_val, result["saving_plan"]['monthly_save'], fixed_expenses_total, variable_expenses_total
                ), use_container_width=True)

        with tab3:
            if st.session_state.analysis_result:
                sensitivity_result = advisor_chain.analyze_sensitivity(
                    result, savings_adjustment, st.session_state.user_input
                )

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(sensitivity_result['sensitivity_chart'], use_container_width=True)
                with col2:
                    st.markdown(f"""
                    ## Sensitivity Analysis

                    ### Current Plan:
                    - **Savings Rate**: {sensitivity_result['sensitivity_analysis']['current_savings_ratio']}%
                    - **Monthly Savings**: ${sensitivity_result['sensitivity_analysis']['current_monthly_save']:,.0f}
                    - **Timeline**: {sensitivity_result['sensitivity_analysis']['current_months']} months

                    ### Adjusted Plan:
                    - **Savings Rate**: {sensitivity_result['sensitivity_analysis']['new_savings_ratio']}%
                    - **Monthly Savings**: ${sensitivity_result['sensitivity_analysis']['new_monthly_save']:,.0f}
                    - **Timeline**: {sensitivity_result['sensitivity_analysis']['new_months']} months

                    ### Impact:
                    - **Time Saved**: {abs(sensitivity_result['sensitivity_analysis']['time_impact'])} months {'faster' if sensitivity_result['sensitivity_analysis']['time_impact'] > 0 else 'slower'}
                    - **Annual Savings Change**: ${sensitivity_result['sensitivity_analysis']['annual_impact']:+,.0f}
                    """)

        with tab4:  # 🆕 SMART FEATURES TAB
            st.markdown("### 🔔 Smart Alerts")
            for alert in result.get('alerts', []):
                st.info(f"{alert['icon']} **{alert['title']}**: {alert['msg']}")
            
            st.markdown("### 👥 Peer Comparison")
            peer = result.get('peer_compare', {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Income Bracket", peer.get('bracket', 'N/A').upper())
                st.metric("Savings Rate", f"{peer.get('user_save_rate', 0)}% vs {peer.get('benchmark_save_rate', 0)}% target")
            with col2:
                st.metric("Savings Status", peer.get('save_status', 'N/A').upper())
                st.metric("Debt Management", peer.get('debt_status', 'N/A').upper())
            
            st.markdown("### 📜 Financial History")
            history = db.get_history()
            if not history.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history['date'], y=history['score'], 
                                        mode='lines+markers', name='Health Score',
                                        line=dict(color='#10b981', width=3)))
                fig.update_layout(
                    title='Health Score Over Time', 
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab5:  # 🆕 CALCULATORS TAB
            st.markdown("### 💰 Emergency Fund Calculator")
            monthly_exp = st.number_input("Monthly Expenses", value=3000, step=100)
            ef = FinCalc.emergency_fund(monthly_exp)
            col1, col2, col3 = st.columns(3)
            col1.metric("3 Months", f"${ef['3_months']:,.0f}")
            col2.metric("6 Months", f"${ef['6_months']:,.0f}")
            col3.metric("12 Months", f"${ef['12_months']:,.0f}")
            
            st.markdown("### 📈 Compound Interest Calculator")
            col1, col2 = st.columns(2)
            with col1:
                principal = st.number_input("Initial Amount", value=10000, step=1000)
                monthly_contrib = st.number_input("Monthly Contribution", value=500, step=50)
            with col2:
                rate = st.number_input("Annual Return %", value=7.0, step=0.5)
                years = st.number_input("Years", value=10, step=1)
            
            fv = FinCalc.compound_interest(principal, monthly_contrib, rate, years)
            st.success(f"### Future Value: ${fv:,.2f}")
            
            st.markdown("### 💳 Debt Payoff Calculator")
            col1, col2, col3 = st.columns(3)
            with col1:
                debt_amt = st.number_input("Debt Amount", value=5000, step=100)
            with col2:
                debt_rate = st.number_input("Interest Rate %", value=18.0, step=0.5)
            with col3:
                if st.button("Add Debt"):
                    st.session_state.debts.append({'balance': debt_amt, 'rate': debt_rate})
            
            if st.session_state.debts:
                st.markdown("#### Current Debts:")
                for i, debt in enumerate(st.session_state.debts):
                    st.write(f"{i+1}. ${debt['balance']:,.0f} at {debt['rate']}%")
                
                if st.button("Clear Debts"):
                    st.session_state.debts = []
                    st.rerun()
                
                payment = st.number_input("Monthly Payment", value=500, step=50)
                if st.button("Calculate Payoff"):
                    result_debt = DebtCalc.snowball(st.session_state.debts.copy(), payment)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Payoff Time", f"{result_debt['months']} months")
                    with col2:
                        st.metric("Total Interest", f"${result_debt['interest']:,.2f}")

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        create_info_card(
            "Get Started",
            "Fill in your financial information in the sidebar and click 'Analyze Financial Status' to receive your personalized financial plan with AI-powered insights and recommendations."
        )

        with st.expander("Usage Examples - أمثلة للاستخدام"):
            st.markdown("""
            **Choose from ready examples or enter your own data:**

            - **Junior Employee**: Income 3000, low savings, goal to buy a car
            - **Middle Class Family**: Income 8000, savings 20000, goal for children's education  
            - **Financial Professional**: Income 15000, savings 50000, goal for early retirement
            - **Home Buyer**: Income 7000, savings 15000, goal for down payment
            - **Student**: Income 2000, savings 1000, goal to pay off student loans
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: var(--text-secondary); font-size: 0.9rem;'>
        <strong>Note</strong>: This financial advisor uses AI to provide general financial advice and is not a substitute for a certified financial planner. 
        Always consult a professional for important financial decisions.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()