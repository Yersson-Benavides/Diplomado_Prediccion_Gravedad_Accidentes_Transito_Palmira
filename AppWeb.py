import dash
from dash import html, dcc
import pandas as pd
import numpy as np
import folium
import plotly.graph_objects as go
import folium.plugins
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dash_bootstrap_components as dbc


df20 = pd.read_csv('datosLimpios2020.csv')
df21 = pd.read_csv('datosLimpios2021.csv')
df22 = pd.read_csv('datosLimpios2022.csv')

df20.drop(columns=['Unnamed: 0'])
df21.drop(columns=['Unnamed: 0'])
df22.drop(columns=['Unnamed: 0'])

UDts = pd.concat([df20, df21, df22])
UDts.drop(columns=['Unnamed: 0'])

# Concatenar los datos en un único DataFrame
data_Frame = pd.concat([df20, df21, df22])

# Obtener las columnas seleccionadas
columnas = ['JORNADA', 'DIA_SEMANA', 'BARRIO-CORREGIMIENTO-VIA', 'GRAVEDAD']
subset = data_Frame[columnas]

# Obtener opciones para la jornada, día de la semana y barrio o corregimiento
opciones_jornada = data_Frame['JORNADA'].unique()
opciones_dia_semana = data_Frame['DIA_SEMANA'].unique()
opciones_barrio_corregimiento_via = data_Frame['BARRIO-CORREGIMIENTO-VIA'].unique()

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

def generate_map(df20, df21, df22):
    feature_gp1 = folium.FeatureGroup(name="2020")
    feature_gp2 = folium.FeatureGroup(name="2021")
    feature_gp3 = folium.FeatureGroup(name="2022")

    m = folium.Map(location=[3.53944, -76.30361], zoom_start=12)

    for index, row in df20.iterrows():
        folium.CircleMarker(location=[row['LAT'], row['LONG']], radius=2, color="red").add_to(feature_gp1)

    for index, row in df21.iterrows():
        folium.CircleMarker(location=[row['LAT'], row['LONG']], radius=2, color="blue").add_to(feature_gp2)

    for index, row in df22.iterrows():
        folium.CircleMarker(location=[row['LAT'], row['LONG']], radius=2, color="black").add_to(feature_gp3)

    feature_gp1.add_to(m)
    feature_gp2.add_to(m)
    feature_gp3.add_to(m)
    folium.LayerControl().add_to(m)

    map_html = m.get_root().render()
    return map_html


def generate_heatmap(UDts):
    mapacalor = folium.Map(location=[3.53944, -76.30361], zoom_start=12)
    datos_mapa = list(zip(UDts['LAT'], UDts['LONG']))
    folium.plugins.HeatMap(datos_mapa).add_to(mapacalor)
    map_html = mapacalor.get_root().render()
    return map_html


a1 = UDts.groupby(['GRAVEDAD', 'AÑO'])['GRAVEDAD'].count()
d1 = a1.unstack()

a2 = UDts.groupby(['JORNADA', 'AÑO'])['JORNADA'].count()
d2 = a2.unstack()

a4 = UDts.groupby(['CLASE_VEHICULO', 'AÑO'])['CLASE_VEHICULO'].count()
d4 = a4.unstack()

a5 = UDts.groupby(['GENERO', 'AÑO'])['GENERO'].count()
d5 = a5.unstack()

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(
                [
                    dcc.Markdown('## Análisis de accidentes de tránsito'),
                    dcc.Markdown('En esta página encontrarás un análisis de los accidentes de tránsito ocurridos en los años 2020, 2021 y 2022, así como una posible predicción para el año 2023 en base a los datos obtenidos.')
                ]
            )
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Mapa de las ubicaciones exactas de los accidentes"),
                                dbc.CardBody(
                                    [
                                        html.Iframe(id='map', srcDoc=generate_map(df20, df21, df22), style={'width': '100%', 'height': '60vh'})
                                    ]
                                ),
                            ]
                        ),
                    ],
                    width=6
                ),

                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Mapa de calor de los accidentes"),
                                dbc.CardBody(
                                    [
                                        html.Iframe(id='heatmap', srcDoc=generate_heatmap(UDts), style={'width': '100%', 'height': '60vh'})
                                    ]
                                ),
                            ]
                        ),
                    ],
                    width=6
                ),
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Gráficas de accidentes por gravedad en cada año"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id='grafica-gravedad',
                                            figure={
                                                'data': [
                                                    go.Bar(x=d1.index, y=d1[2020], name='2020'),
                                                    go.Bar(x=d1.index, y=d1[2021], name='2021'),
                                                    go.Bar(x=d1.index, y=d1[2022], name='2022')
                                                ],
                                                'layout': go.Layout(
                                                    title='Accidentes por gravedad en cada año',
                                                    xaxis={'title': 'Gravedad'},
                                                    yaxis={'title': 'Cantidad de accidentes'},
                                                    barmode='group'
                                                )
                                            }
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                    width=6
                ),

                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Gráficas de accidentes por jornada en cada año"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id='grafica-jornada',
                                            figure={
                                                'data': [
                                                    go.Bar(x=d2.index, y=d2[2020], name='2020'),
                                                    go.Bar(x=d2.index, y=d2[2021], name='2021'),
                                                    go.Bar(x=d2.index, y=d2[2022], name='2022')
                                                ],
                                                'layout': go.Layout(
                                                    title='Accidentes por jornada en cada año',
                                                    xaxis={'title': 'Jornada'},
                                                    yaxis={'title': 'Cantidad de accidentes'},
                                                    barmode='group'
                                                )
                                            }
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                    width=6
                ),
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Gráficas de accidentes por clase de vehículo en cada año"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id='grafica-vehiculo',
                                            figure={
                                                'data': [
                                                    go.Bar(x=d4.index, y=d4[2020], name='2020'),
                                                    go.Bar(x=d4.index, y=d4[2021], name='2021'),
                                                    go.Bar(x=d4.index, y=d4[2022], name='2022')
                                                ],
                                                'layout': go.Layout(
                                                    title='Accidentes por clase de vehículo en cada año',
                                                    xaxis={'title': 'Clase de vehículo'},
                                                    yaxis={'title': 'Cantidad de accidentes'},
                                                    barmode='group'
                                                )
                                            }
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                    width=6
                ),

                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Gráficas de accidentes por género en cada año"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(
                                            id='grafica-genero',
                                            figure={
                                                'data': [
                                                    go.Bar(x=d5.index, y=d5[2020], name='2020'),
                                                    go.Bar(x=d5.index, y=d5[2021], name='2021'),
                                                    go.Bar(x=d5.index, y=d5[2022], name='2022')
                                                ],
                                                'layout': go.Layout(
                                                    title='Accidentes por género en cada año',
                                                    xaxis={'title': 'Género'},
                                                    yaxis={'title': 'Cantidad de accidentes'},
                                                    barmode='group'
                                                )
                                            }
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                    width=6
                ),
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("A continuación puedes acceder a un modelo de predicción haciendo clic en el botón de abajo"),
                                dbc.CardBody(
                                    [
                                        dbc.Button('Ir a predicción', id='btn-prediction', n_clicks=0),
                                        html.Div(id='prediction-modal')
                                    ]
                                ),
                            ]
                        ),
                    ]
                )
            ]
        )
    ],
    className='container-fluid',
    style={'padding': '50px','backgroundColor': 'white'}
)

@app.callback(
    Output('prediction-modal', 'children'),
    [Input('btn-prediction', 'n_clicks')]
)

def open_prediction_modal(n_clicks):
    if n_clicks:
        return dbc.Modal(
            [
                dbc.ModalHeader("Realizar predicciones de accidentes de tránsito"),
                dbc.ModalBody(
                    [
                        html.P('Aquí puedes realizar predicciones de accidentes de tránsito utilizando un modelo predictivo.'),
                        # Prediccion arbol de desición
                        html.H1("Realizar predicciones de accidentes de tránsito"),
                        html.Div(
                            [
                                html.P('Jornada:'),
                                dcc.Dropdown(
                                    id='jornada-dropdown',
                                    options=[{'label': opcion, 'value': opcion} for opcion in opciones_jornada],
                                    placeholder="Seleccione una jornada"
                                )
                            ],
                            style={'width': '30%', 'display': 'inline-block'}
                        ),
                        html.Div(
                            [
                                html.P('Día de la semana:'),
                                dcc.Dropdown(
                                    id='dia-semana-dropdown',
                                    options=[{'label': opcion, 'value': opcion} for opcion in opciones_dia_semana],
                                    placeholder="Seleccione un día de la semana"
                                )
                            ],
                            style={'width': '30%', 'display': 'inline-block','color': 'black'}
                        ),
                        html.Div(
                            [
                                html.P('Barrio o corregimiento:'),
                                dcc.Dropdown(
                                    id='barrio-corregimiento-dropdown',
                                    options=[{'label': opcion, 'value': opcion} for opcion in opciones_barrio_corregimiento_via],
                                    placeholder="Seleccione un barrio o corregimiento"
                                )
                            ],
                            style={'width': '30%', 'display': 'inline-block'}
                        ),
                        dbc.Button('Realizar predicción', id='submit-button', n_clicks=0),
                        html.Div(id='prediction-output')
                    ]
                ),
                dbc.ModalFooter(
                    dbc.Button("Cerrar", id="close-prediction", className="ml-auto")
                ),
            ],
            id="modal",
            size='lg'
        )
# Definir el callback para realizar la predicción
@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('jornada-dropdown', 'value'),
    Input('dia-semana-dropdown', 'value'),
    Input('barrio-corregimiento-dropdown', 'value')
)
def make_prediction(n_clicks, jornada, dia_semana, barrio_corregimiento_via):
    if n_clicks:
        if jornada not in opciones_jornada or dia_semana not in opciones_dia_semana or barrio_corregimiento_via not in opciones_barrio_corregimiento_via:
            return html.P("No has ingresado una opción válida. Por favor, inténtalo nuevamente.")
        
        # Preparar los datos del usuario
        user_data = pd.DataFrame({
            'JORNADA': [jornada],
            'DIA_SEMANA': [dia_semana],
            'BARRIO-CORREGIMIENTO-VIA': [barrio_corregimiento_via]
        })
        
        # Codificar las variables categóricas en el conjunto de datos completo
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        data_Frame_encoded = encoder.fit_transform(data_Frame[['JORNADA', 'DIA_SEMANA', 'BARRIO-CORREGIMIENTO-VIA']])
        
        # Obtener las columnas correspondientes a los datos del usuario
        user_data_encoded = encoder.transform(user_data)
        
        # Dividir los datos en características (X) y objetivo de predicción (y)
        X = data_Frame_encoded
        y = LabelEncoder().fit_transform(data_Frame['GRAVEDAD'])
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar el modelo Árbol de Decisión
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        
        # Realizar la predicción con los datos del usuario
        prediction = model.predict(user_data_encoded)
        
        # Inicializar un nuevo codificador de etiquetas y ajustarlo con los datos de gravedad
        gravity_encoder = LabelEncoder()
        gravity_encoder.fit(data_Frame['GRAVEDAD'])
        
        # Obtener la etiqueta correspondiente a la predicción de gravedad
        predicted_gravity = gravity_encoder.inverse_transform(prediction)
        
        # Calcular la precisión del modelo en el conjunto de prueba
        y_pred = model.predict(X_test)
        precision = accuracy_score(y_test, y_pred)
        precision_percentage = precision * 100
        
        return html.Div([
            html.P("Predicción de gravedad: {}".format(predicted_gravity[0])),
            html.P("Precisión del modelo: {:.2f}%".format(precision_percentage))
        ])

@app.callback(
    Output('modal', 'is_open'),
    [Input('btn-prediction', 'n_clicks'),
     Input('close-prediction', 'n_clicks')],
    [State('modal', 'is_open')],
)
def toggle_prediction_modal(n_clicks, close_clicks, is_open):
    if n_clicks or close_clicks:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=True)
    