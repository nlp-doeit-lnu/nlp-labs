import os
import logging
import math
import random
import multiprocessing
import traceback
import webbrowser as web
import concurrent.futures
import codecs

import numpy as np
import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

import dash_html_components as html
import dash_core_components as dcc
import dash_table as dt


needed_chars = []
charArr = []
symbArr = []

currChar = 0

while len(charArr) < 45000:
    currChar += 1
    if (chr(currChar).isalpha()):
        charArr.append(chr(currChar))

currSymb = 0

tempArr = []

while len(tempArr) < 1001:
    currSymb += 1
    if (not(chr(currSymb).isalpha()) and chr(currSymb).isprintable()):
        tempArr.append(chr(currSymb))

symbArr = tempArr[16:26]+tempArr[1:16] +tempArr[26:]

needed_chars = charArr + symbArr


red_inline_border_style = {"margin-left": "2%", "border": "2px solid red", "width": "40%", "display": "inline-block"}
red_options_border_style = {"margin-right": "2%", "border": "2px solid red"}
default_inline_style = {"margin-left": "2%", "width": "40%", "display": "inline-block"}
default_options_style = {"margin-right": "2%", "float": "left"}

CLICKS_GENERATE_BTN = 0
IS_IN_GENERATING = False

logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Monkey Text Generator'
app.layout = html.Div([
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Main generation parameters"),
                        dbc.CardBody(
                            [
                                html.Label("M"),
                                dbc.Input(id='M', type="number",
                                          style={"margin-left": "2%", "margin-right": "2%", "width": "40%", "display": "inline-block"}),
                                html.Label("L"),
                                dbc.Input(id='l', type="number", min="1",
                                          style={"margin-left": "2%", "width": "40%", "display": "inline-block"}),
                                html.Br(),
                                html.Label("f0"),
                                dbc.Input(id='f0', type="number", value='0',
                                          style={"margin-left": "2%", "width": "40%", "display": "inline-block"})
                            ]
                        )
                    ]
                ), width=3
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Options"),
                        dbc.CardBody(
                            [
                                html.Div(
                                    dcc.RadioItems(
                                        options=[
                                            {"label": "equal", "value": "eq"},
                                            {"label": "random", "value": "rand"},
                                            {"label": "barrier", "value": "bar"},
                                            {"label": "log", "value": "log"},
                                        ],
                                        id="options",
                                        value="eq"
                                    ), style={"margin-right": "2%", "float": "left"}
                                ),
                                html.Div(
                                    [
                                        dbc.Input(id='C', min="1", type="number", placeholder="C:barrier", value="",
                                                  style={"height": "40%"}),
                                        dbc.Input(id='A', type="number", placeholder="A:log", value="", style={"height": "40%"}, step=0.0001),
                                        html.Div(id='A-is-wrong-message')
                                    ], style={"float": "left", "width": "30%"}
                                )
                            ]
                        )
                    ]
                ), width=5
            ),
            dbc.Col(
                html.Div(
                    [
                        dbc.Button(
                            "Generate & Save",
                            id="generate_save_btn",
                            style={"margin-top": "45%"},
                            disabled=True),
                        dcc.Checklist(
                            id='multiprocess_computing',
                            options=[
                                {'label': 'multiprocessing', 'value': 'enable'}
                            ],
                            value=''
                        )
                    ],
                    style={"float": "left", "width": "50%"}
                ), width=3)
        ]
    )
    ,
    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                dt.DataTable(
                    id="table",
                    editable=True,
                    sort_action="native",
                    style_cell={
                      'textAlign': 'center',
                      'whiteSpace': 'auto',
                      'height': 'auto',
                      'width': 'auto',
                      'minWidth': 95,
                      'maxWidth': 95
                    },
                    fixed_rows={"headers": True},
                    style_table={"height": "80%", "overflowY": "auto"},
                    columns=[
                      {"name": column, "id": column}
                      for column in ["rank", "unicode", "character", "P-theory", "P-experiment"]
                    ]
                ), width=6
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Preview 1000 elements according to the set parameters"),
                        dbc.CardBody(
                            dcc.Textarea(
                                id="text",
                                placeholder="",
                                value="",
                                style={"height": "100%", "width": "100%"},
                                rows=19
                            )
                        )
                    ]
                ), width=6
            )
        ]
    ),
    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Checklist(
                        id='enable_histogram',
                        options=[
                            {'label': 'enable histogram', 'value': 'enable'}
                        ],
                        value=''
                    ),
                    dcc.RadioItems(
                        id='hist_option',
                        options=[
                           {"label": "linear", "value": "lin"},
                           {"label": "log", "value": "log"}
                        ],
                        value="lin"
                    ),
                    dcc.Graph(
                        id="histogram"
                    )
                ], width=12
            )
        ]
    )
], style={"padding": "1%"})


@app.callback(
    Output("histogram", "figure"),
    [
        Input("table", "derived_virtual_data"),
        Input("hist_option", "value"),
        Input("enable_histogram", "value")
    ]
)
def update_histogram(table, option, enable):
    if enable != ['enable'] or not table:
        return {}
    if option == "lin":
        log = False
    else:
        log = True
    df = pd.DataFrame(data=table)
    df = df[df["character"] != "space"]
    fig = px.bar(df, x="rank", y="P-experiment",
                 barmode='overlay', hover_data=["character"], labels={"value": "probability"}, log_x=log)
    fig = px.bar(df, x="rank", y=["P-theory", "P-experiment"],
                 barmode='overlay', color_discrete_sequence=['yellow', 'blue'],
                 labels={"value": "probability"}, hover_data=["character"])

    import plotly.graph_objects as go
    df_theory = df[["rank", "P-theory"]]
    df_experiment = df[["rank", "P-experiment"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_theory["rank"], y=df_theory["P-theory"], mode="lines", name='Theory', line={"color": "red"}))
    fig.add_trace(go.Bar(x=df_experiment["rank"], y=df_experiment["P-experiment"], name='Experiment', marker_color='green'))
    fig.update_layout(
        barmode='group',
        xaxis_type='log' if log else 'linear',
        xaxis_title='Rank',
        yaxis_title='Probability',
        title='Relative frequencies of characters in a sequence'
    )
    return fig


@app.callback(
    Output("generate_save_btn", "disabled"),
    [
        Input("l", "value"),
        Input("M", "value")
    ]
)
def generation_valid(l, M):
    global NEED_TO_GENERATE_FILE
    if not l or not M:
        return True
    else:
        NEED_TO_GENERATE_FILE = False
        return False


# @app.callback(
#     Input("generate_save_btn", "n_clicks")
# )
# def click_generate_btn(n_clicks):
#     global NEED_TO_GENERATE_FILE
#     NEED_TO_GENERATE_FILE = True


def process_function(characters, probabilities, length):
    result = list()
    previous_symbol = ' '
    iteration = 0
    while iteration < int(length):
        # генерація нового елемента послідовності
        current_symbol = random.choices(characters, weights=probabilities, k=1)[0]
        # видалення пробілу, якщо він є дублем
        if current_symbol == previous_symbol == " ":
            continue
        # додавання згенерованого символу в кінець послідовності
        result.append(current_symbol)
        previous_symbol = current_symbol
        iteration += 1
    return result


def generate_and_save(M, l, table, options, C, A, f0, enable_mp):
    global IS_IN_GENERATING
    IS_IN_GENERATING = True
    # generating
    if float(f0) != 0:
        table[0]["character"] = " "

    df = pd.DataFrame(table)

    text = list()
    if (enable_mp):
        # concurate
        cc = multiprocessing.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=cc) as executor:
            futures = []
            for i in range(cc):
                futures.append(
                    executor.submit(
                        process_function,
                        df.character,
                        df["P-theory"],
                        int(float(l) / float(cc))
                    )
                )  # make length int()
            for future in concurrent.futures.as_completed(futures):
                part_text = future.result()
                text = [*text, *part_text]
    else:
        # sequential
        text = process_function(characters=df.character, probabilities=df["P-theory"], length=int(l))

    df["P-experiment"] = df.character.apply(
        lambda symbol: round(text.count(symbol) / int(l), 7)
    )

    if float(f0) != 0:
        df.iloc[0]["character"] = "space"

    # saving
    name = f"M={M}, L={l}, space={f0}, {options}"
    if options == "bar":
        name += f" C={C}"
    if options == "log":
        name += f" A={A}"


    curr_dirname = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(f'{curr_dirname}/output'):
        os.mkdir(f'{curr_dirname}/output')

    df.to_csv(f"{curr_dirname}/output/{name}--statistics.txt", index=False, sep='\t')

    file = codecs.open(f'{curr_dirname}/output/{name}--text.txt', "w", "utf-8-sig")
    file.write("".join(text))
    file.close()

    return text


@app.callback(
    [
        Output("table", "data"),
        Output("text", "value"),
        Output("C", "style"),
        Output("A", "style"),
        Output("M", "style"),
        Output("f0", "style"),
        Output(component_id='A-is-wrong-message', component_property='children')
    ],
    [
        Input('M', 'value'),
        Input('f0', 'value'),
        Input("options", "value"),
        Input("C", "value"),
        Input("A", "value"),
        Input("generate_save_btn", "n_clicks")
    ],
    [
        State("l", "value"),
        State("multiprocess_computing", "value")
    ]
)
def update_m(M, f0, options, C, A, n_clicks, l, mp): # table.data, text.value, C.style, A.style, M.style, f0.style, A.children
    data = {"rank": [], "unicode": [], "character": [], "probability": []}
    prob_accuracy = 7
    try:
        M = int(M)
        if  M < 1:
            raise Exception("M < 1 error")
    except:
        return None, "", default_options_style, default_options_style, red_inline_border_style, default_inline_style, None

    try:
        f0 = float(f0) if f0 is not None else 0.0
        if not (0.0 <= f0 <= 1.0):
            raise Exception("not 0 ≤ f0 ≤ 1 error")
    except:
        return [], dash.no_update, default_options_style, default_options_style, default_inline_style, red_inline_border_style, None

    if options == "rand":
        ri = np.random.uniform(0, 1, size=M)
        s = sum(ri)
        li = ri / s * (1 - f0)
        li = [round(element, prob_accuracy) for element in li]
        data['probability'] = [f0] + li

    if options == "eq":
        li = []
        li.append(f0)
        data["probability"] = li + [round((1.0 - f0) / (M), prob_accuracy) for _ in range(M)]

    if options == "bar":
        try:
            C = float(C)
        except:
            return [], dash.no_update, red_options_border_style, default_options_style, default_inline_style, default_inline_style, None

        a = ((C - 1.0) / (C + 1.0)) * ((2.0 * (1.0 - f0)) / (M * (M - 1.0)))
        f1 = (2.0 * C * (1.0 - f0)) / (M * (C + 1.0))

        li = []
        li.append(f0)
        li.append(round(f1, prob_accuracy))

        f = f1
        for j in range(M - 1):
            f -= a
            li.append(round(f, prob_accuracy))

        data["probability"] = li

    if options == "log":
        log10_Mfactorial = 0
        for j in range(int(M)):
            log10_Mfactorial += math.log10(j + 1)
        min_A = round((1 - float(f0)) / float(M), 5)
        max_A = round((1 - float(f0)) / (float(M) - log10_Mfactorial / math.log10(float(M))), 5)
        try:
            A = float(A)
            if not (min_A <= A <= max_A):
                raise Exception()
        except:
            return [], dash.no_update, default_options_style, red_options_border_style, default_inline_style, default_inline_style, html.Div(f'A: {min_A} ≤ A ≤ {max_A}')

        # log10_Mfactorial=math.log10(math.factorial(int(M))) #simplify factorial!
        log10_Mfactorial = 0
        for j in range(M):
            log10_Mfactorial += math.log10(j + 1)

        B = (M * A - (1.0 - f0)) / log10_Mfactorial

        li = []
        li.append(f0)
        for j in range(M):
            f = A - (B * math.log10(j + 1))
            li.append(round(f, prob_accuracy))

        data["probability"] = li

    k = 0
    i = 32
    while len(data["character"]) != (M + 1):
        data['character'].append(needed_chars[i - 33])  # char generator
        data['unicode'].append(ord(needed_chars[i - 33]))
        data["rank"].append(i - 32 - k)
        i += 1

    data['character'][0] = "space"
    if (float(f0) == 0):
        data['character'].pop(0)
        data['unicode'].pop(0)
        data["rank"].pop(0)
        data["probability"].pop(0)
    else:
        data['unicode'][0] = 32
        data['character'][0] = " "

    data = pd.DataFrame(data=data)
    data.rename(columns={"probability": "P-theory"}, inplace=True)

    global CLICKS_GENERATE_BTN, IS_IN_GENERATING
    if n_clicks and not IS_IN_GENERATING and n_clicks > CLICKS_GENERATE_BTN and l:
        text_list = generate_and_save(M, l, data.to_dict("records"), options, C, A, f0, mp)
        CLICKS_GENERATE_BTN = n_clicks
    else:
        text_list = process_function(data.character, probabilities=data["P-theory"], length=10_000)
        IS_IN_GENERATING = False

    text_1000_str = "".join(text_list[:1000])

    data["P-experiment"] = data.character.apply(
        lambda symbol: round(text_list.count(symbol) / len(text_list), prob_accuracy))

    if float(f0) > 0:
        data.character = data.character.apply(lambda letter: "space" if letter == " " else letter)

    return data.to_dict("records"), text_1000_str, default_options_style, default_options_style, default_inline_style, default_inline_style, None


if __name__ == "__main__":
    # web.open("http://127.0.0.1:8050/")
    web.open("http://0.0.0.0:8050/")
    app.run_server(debug=False, host="0.0.0.0", port=8050)
