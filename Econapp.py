

import pandas as pd
import numpy as np
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from sklearn.ensemble import RandomForestRegressor

# =====================
# 1. DATA PREPARATION
# =====================
countries = ['USA', 'China', 'UK', 'Russia', 'UAE', 'Iran', 'Germany', 'India', 'Brazil']
np.random.seed(42)

data = {
    'Year': np.random.choice(range(2010, 2023), 1000),
    'Country': np.random.choice(countries, 1000),
    # Traditional factors
    'Tech_Investment': np.random.uniform(0.5, 5, 1000),
    'Education_Spending': np.random.uniform(1, 8, 1000),
    'Environmental_Protection': np.random.uniform(30, 95, 1000),
    # Modern factors
    'Urbanization_Rate': np.random.uniform(30, 95, 1000),
    'Trade_Openness': np.random.uniform(10, 150, 1000),  # Globalization proxy
    'Individualism_Index': np.random.uniform(10, 90, 1000),  # Hofstede scale
    # Outcomes
    'GDP_Growth': np.random.normal(3, 1.5, 1000)
}

df = pd.DataFrame(data)

# Add realistic relationships
df['GDP_Growth'] = (
    df['GDP_Growth'] 
    + 0.12 * df['Tech_Investment']
    + 0.08 * df['Education_Spending']
    + 0.05 * df['Environmental_Protection']
    + 0.07 * df['Urbanization_Rate']
    + 0.04 * df['Trade_Openness']
    + 0.03 * df['Individualism_Index']
)

# Country-specific adjustments
adjustments = {
    'USA': {'Individualism_Index': +15, 'Tech_Investment': +2},
    'UAE': {'Environmental_Protection': -10, 'Trade_Openness': +40},
    'Iran': {'Trade_Openness': -30, 'Tech_Investment': -1},
    'Russia': {'Individualism_Index': -10, 'Education_Spending': -0.5}
}

for country, adj in adjustments.items():
    mask = df['Country'] == country
    for col, val in adj.items():
        df.loc[mask, col] += val

# =====================
# 2. MODEL TRAINING
# =====================
X = pd.get_dummies(df.drop(['GDP_Growth', 'Country'], axis=1))
y = df['GDP_Growth']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =====================
# 3. DASH APP
# =====================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Global Economic Prosperity Simulator"), className="text-center my-4")),
    
    dbc.Row(
        dbc.Col(
            html.Div(
                "Done By Ronak Baniabbasi | May 5, 2025",
                style={
                    'textAlign': 'center',
                    'color': 'gray',
                    'marginTop': '30px',
                    'fontSize': '0.9rem'
                }
            ),
            width=12
        )
    ),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Country:"),
            dcc.Dropdown(id='country', options=[{'label':c, 'value':c} for c in countries], value='China')
        ], md=4),
        dbc.Col([
            html.Label("Projection Years:"),
            dcc.Slider(id='years', min=1, max=10, value=5, marks={i: str(i) for i in range(1, 11)})
        ], md=8)
    ]),
    
    
    dbc.Tabs([
        dbc.Tab(label="Traditional Factors", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Technology Investment Change (%):"),
                    dcc.Slider(id='tech', min=-10, max=50, value=0)
                ]),
                dbc.Col([
                    html.Label("Education Spending Change (%):"),
                    dcc.Slider(id='edu', min=-10, max=30, value=0)
                ])
            ]),
            dbc.Row(dbc.Col([
                html.Label("Environmental Protection Change (%):"),
                dcc.Slider(id='env', min=-20, max=40, value=0)
            ]))
        ]),
        
        dbc.Tab(label="Modern Factors", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Urbanization Rate Change (%):"),
                    dcc.Slider(id='urban', min=-5, max=25, value=0)
                ]),
                dbc.Col([
                    html.Label("Globalization (Trade Openness) Change (%):"),
                    dcc.Slider(id='global', min=-30, max=40, value=0)
                ])
            ]),
            dbc.Row(dbc.Col([
                html.Label("Individualism Index Change (%):"),
                dcc.Slider(id='individual', min=-10, max=20, value=0)
            ]))
        ])
    ], className="my-4"),
    
    dbc.Card([
        dbc.CardHeader("Simulation Results", className="h4"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Div(id='gdp-impact', className="h3 text-center")),
                dbc.Col(html.Div(id='factor-analysis'))
            ]),
            dcc.Graph(id='importance-plot')
        ])
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id='country-comparison'), width=8),
        dbc.Col(dcc.Graph(id='factor-tradeoff'), width=4)
    ], className="mt-4")
])


@app.callback(
    [Output('gdp-impact', 'children'),
     Output('factor-analysis', 'children'),
     Output('importance-plot', 'figure'),
     Output('country-comparison', 'figure'),
     Output('factor-tradeoff', 'figure')],
    [Input('country', 'value'),
     Input('tech', 'value'),
     Input('edu', 'value'),
     Input('env', 'value'),
     Input('urban', 'value'),
     Input('global', 'value'),
     Input('individual', 'value'),
     Input('years', 'value')]
)
def update_dashboard(country, tech, edu, env, urban, global_, individual, years):
    # Scale policy impacts by time horizon
    scale = 1 + (years / 10)
    inputs = {
        'Tech_Investment': tech * scale,
        'Education_Spending': edu * scale,
        'Environmental_Protection': env * scale,
        'Urbanization_Rate': urban * scale,
        'Trade_Openness': global_ * scale,
        'Individualism_Index': individual * scale,
        'Year': 2023 + years
    }
    
    # Predict GDP impact
    input_df = pd.DataFrame([inputs])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
    gdp_change = model.predict(input_df)[0] - df[df['Country'] == country]['GDP_Growth'].mean()
    
    # Create outputs
    gdp_result = html.Div([
        html.H4(f"Projected GDP Change in {years} Years:"),
        html.H2(f"{gdp_change:.2f}%", style={'color': 'green' if gdp_change > 0 else 'red'})
    ])
    
    factor_analysis = html.Ul([
        html.Li(f"Technology: {tech}%"),
        html.Li(f"Education: {edu}%"),
        html.Li(f"Environment: {env}%"),
        html.Li(f"Urbanization: {urban}%"),
        html.Li(f"Globalization: {global_}%"),
        html.Li(f"Individualism: {individual}%")
    ])
    
    # Visualizations
    importance = pd.DataFrame({'Factor': X.columns, 'Importance': model.feature_importances_})
    importance_fig = px.bar(importance, x='Factor', y='Importance', title='Factor Importance for GDP Growth')
    
    country_comparison = px.line_polar(
        df.groupby('Country').mean().reset_index(),
        r='GDP_Growth',
        theta='Country',
        title='GDP Growth Comparison'
    )
    
    tradeoff_fig = px.scatter(
        df,
        x='Individualism_Index',
        y='Environmental_Protection',
        color='GDP_Growth',
        hover_name='Country',
        title='Individualism vs Environmental Protection Trade-offs'
    )
    
    return gdp_result, factor_analysis, importance_fig, country_comparison, tradeoff_fig


    # Your name and date added here
    


if __name__ == '__main__':
    app.run( port=8000)
