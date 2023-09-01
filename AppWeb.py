import dash
from dash import html, dcc
import pandas as pd
import numpy as np
import folium
import plotly.graph_objects as go
import folium.plugins
from dash.dependencies import Input, Output, State
from dash import html

import dash_bootstrap_components as dbc

# from dash import Input, Output, State
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.express as px

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

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

# Concatenar los datos en un único DataFrame
data_Frame = pd.concat([df20, df21, df22])

# Preprocesamiento de datos
features = ['JORNADA', 'DIA_SEMANA', 'BARRIO-CORREGIMIENTO-VIA', 'LAT', 'LONG', 'GRAVEDAD']
filtered_data = data_Frame[features].copy()

# División de datos para modelos de regresión y clasificación
regression_features = ['JORNADA', 'DIA_SEMANA', 'BARRIO-CORREGIMIENTO-VIA']
classification_features = ['JORNADA', 'DIA_SEMANA', 'BARRIO-CORREGIMIENTO-VIA']

regression_data = filtered_data[regression_features + ['LAT', 'LONG']]
classification_data = filtered_data[classification_features + ['GRAVEDAD']]

# Preparar datos para modelos
X_regression = pd.get_dummies(regression_data.drop(['LAT', 'LONG'], axis=1))
y_latitude = regression_data['LAT']
y_longitude = regression_data['LONG']

X_classification = pd.get_dummies(classification_data.drop('GRAVEDAD', axis=1))
y_gravity = classification_data['GRAVEDAD']

# División en conjuntos de entrenamiento y prueba para modelos de regresión
Xr_train, Xr_test, yr_lat_train, yr_lat_test = train_test_split(X_regression, y_latitude, test_size=0.2, random_state=42)
yr_long_train, yr_long_test = train_test_split(y_longitude, test_size=0.2, random_state=42)

# Entrenamiento de modelos de regresión lineal
regression_model_lat = LinearRegression()
regression_model_long = LinearRegression()

classification_model = RandomForestClassifier(random_state=42)

# Entrenar modelos de regresión lineal
regression_model_lat.fit(Xr_train, yr_lat_train)
regression_model_long.fit(Xr_train, yr_long_train)

# Entrenar modelo de clasificación
classification_model.fit(X_classification, y_gravity)

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Obtener opciones únicas para los dropdowns
jornada_options = [{'label': jornada, 'value': jornada} for jornada in df20['JORNADA'].unique()]
dia_semana_options = [{'label': dia, 'value': dia} for dia in df20['DIA_SEMANA'].unique()]
ubicacion_options = [{'label': ubicacion, 'value': ubicacion} for ubicacion in df20['BARRIO-CORREGIMIENTO-VIA'].unique()]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP, "/assets/style.css"])
server = app.server

# Función para generar el mapa de Folium
def generate_folium_map(map_data):
    # Crear un mapa de Folium
    m = folium.Map(location=[map_data['LAT'].iloc[0], map_data['LONG'].iloc[0]], zoom_start=15)

    # Iterar a través de los datos reales y agregar marcadores al mapa
    for _, row in map_data.iterrows():
        lat = row['LAT']
        lon = row['LONG']
        gravedad = row['GRAVEDAD']
        
        # Personalizar el color del marcador según la gravedad
        if gravedad == 'MUERTOS':
            color = 'red'
        elif gravedad == 'HERIDOS':
            color = 'blue'
        else:
            color = 'green'
        
        # Crear una leyenda
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 1px solid gray; border-radius: 5px;">
            <div style="background-color: red; width: 20px; height: 20px; display: inline-block;"></div><span style="padding-left: 5px;">MUERTOS</span><br>
            <div style="background-color: blue; width: 20px; height: 20px; display: inline-block;"></div><span style="padding-left: 5px;">HERIDOS</span><br>
            <div style="background-color: green; width: 20px; height: 20px; display: inline-block;"></div><span style="padding-left: 5px;">DAÑOS</span><br>
        </div>
        '''
        
        # Agregar la leyenda al mapa
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Crear un marcador con información de la gravedad y agrégalo al mapa
        folium.Marker([lat, lon], tooltip=gravedad, icon=folium.Icon(color=color)).add_to(m)

    # Convierte el mapa de Folium a HTML
    map_html = m.get_root().render()

    return html.Div([html.Iframe(srcDoc=map_html, width='100%', height='600')])

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


#Grafico frecuencia de accidentes

# Paso 1: Cargar y preparar los datos
UDts3 = UDts.copy()
UDts3["HORA"] = UDts3["HORA"].replace({'20:00:00 p: m:':'20:00'}, regex=True)
UDts3["HORA"] = UDts3["HORA"].replace({'NOCHE':'20:00'}, regex=True)
UDts3["HORA"] = UDts3["HORA"].replace({'1900-08-17T00:00':'20:00'}, regex=True)
unique_dates = set(UDts3['FECHA'])
unique_times = set(UDts3['HORA'])

# Paso 2: Combinar las columnas de fecha y hora
UDts3['FECHAHORA'] = pd.to_datetime(UDts3["FECHA"] + " " + UDts3["HORA"])

# Paso 3: Agrupación temporal
UDts3 = UDts3.set_index('FECHAHORA')
data_daily = UDts3.resample('D').size()

# Crear figuras para las gráficas
figs = []

for year in [2020, 2021, 2022]:
    data_year = data_daily[data_daily.index.year == year]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_year.index, y=data_year.values, mode='lines', name=f'Año {year}'))
    fig.update_layout(title=f'Serie temporal de accidentes de tráfico - {year}',
                      xaxis_title='Fecha', yaxis_title='Número de accidentes')
    figs.append(fig)


#Grafico pastel Gravedad, Jornada y Año

UDts4 = UDts.copy()
data_dict = {}
for year in UDts4['AÑO'].unique():
    data_dict[year] = UDts4[UDts4['AÑO'] == year]['GRAVEDAD'].value_counts()

data_df = pd.DataFrame(data_dict)

# Diseño de la interfaz de usuario
app.layout = html.Div([
    html.Link(rel='stylesheet', href='/assets/style.css'),
    dbc.Navbar(
        children=[
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/logo.png", height="50px")),
                        dbc.Col(dbc.NavbarBrand("Diplomado en Ciencia de Datos 'Tomando decisiones basadas en conocimiento'", className="ml-2")),
                    ],
                    align="center",
                ),
                href="/",
            )
        ],
        className=' NavbarFondo',
        style={'fontWeight': 'bold', 'fontWeight': '600', 'backgroundColor': 'rgba(255, 255, 255, 0.5)'}
    ),
    dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                dbc.Col(
                    [
                        dcc.Markdown('# APLICACIÓN DE LA CIENCIA DE DATOS EN EL DESARROLLO DE UN PRODUCTO DE DATOS, PARA LA ESTIMACIÓN DE LA GRAVEDAD DE UN ACCIDENTE DE TRANSITO POR BARRIO, JORNADA Y DÍA QUE PERMITA MEJORAR LA TOMA DE DECISIONES EN RELACIÓN CON LAS ESTRATEGIAS DE PREVENCIÓN DENTRO DEL MUNICIPIO DE PALMIRA'),
                        dcc.Markdown('##### En este aplicativo web, se presenta un detallado análisis de los incidentes de tráfico registrados en los años 2020, 2021 y 2022. Además, se ofrece la posibilidad de realizar predicciones sobre la gravedad de estos incidentes, tomando en consideración variables cruciales como el lugar (barrio), la franja horaria (jornada) y el día de la semana en que tuvo lugar el accidente. Este enfoque analítico brinda una visión integral de los patrones y tendencias asociados a los accidentes de tránsito, al tiempo que permite anticipar la severidad de los mismos en función de variables clave.', style={'marginTop':'50px'})
                    ],
                    className= 'FondoTitulos'
                )
            ),
        ],
        className = 'fondo container-fluid'
    ),
    dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("A continuación puedes acceder a un modelo de predicción haciendo clic en el botón de abajo"),
                                    dbc.CardBody(
                                        [
                                            html.H1("Predicción de Accidentes de Tránsito"),
                                            html.Label("Selecciona la Jornada:"),
                                            dcc.Dropdown(
                                                id='jornada-dropdown',
                                                options=jornada_options,
                                                value=jornada_options[0]['value']
                                            ),
                                            html.Label("Selecciona el Día de la Semana:"),
                                            dcc.Dropdown(
                                                id='dia-semana-dropdown',
                                                options=dia_semana_options,
                                                value=dia_semana_options[0]['value']
                                            ),
                                            html.Label("Selecciona la Ubicación:"),
                                            dcc.Dropdown(
                                                id='ubicacion-dropdown',
                                                options=ubicacion_options,
                                                value=ubicacion_options[0]['value']
                                            ),
                                            dbc.Button('Predecir', className= 'ButtonPredict', id='predict-button'),
                                            html.Div(id='prediction-output'),
                                            html.Div(id='accidents-map')  # Placeholder para el mapa de Folium
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    )
                ]
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
                                    dbc.CardHeader("Gráfico de accidentes de tránsito por año"),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(figure=fig) for fig in figs
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
                            dbc.Card(
                                [
                                    dbc.CardHeader("Gráfico de gravedad Vs Jornada por año"),
                                    dbc.CardBody(
                                        [
                                            dcc.Dropdown(
                                                id='year-dropdown',
                                                options=[{'label': year, 'value': year} for year in data_df.index],
                                                value=data_df.index[0]
                                            ),
                                            dcc.Graph(id='pie-chart'),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=6
                    ),
                ]
            ),
        ],
        className="container-fluid",
        style={"padding": "50px", "backgroundColor": "white"},
    ),
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(
                        [
                            html.A(
                                [
                                    html.Img(src="/assets/linkedin.png", width="30px"),
                                    html.Div(
                                        [
                                            html.P("Linkedin", className="social-media-text"),
                                        ],
                                        className="social-media-icon-text"
                                    )
                                ],
                                href="https://www.linkedin.com/in/yersson-benavides/",
                                target="_blank",
                                className="social-media-icon-link"
                            ),
                            html.A(
                                [
                                    html.Img(src="/assets/github.png", width="30px"),
                                    html.Div(
                                        [
                                            html.P("GitHub", className="social-media-text"),
                                        ],
                                        className="social-media-icon-text"
                                    )
                                ],
                                href="https://github.com/Yersson-Benavides/Diplomado_Prediccion_Gravedad_Accidentes_Transito_Palmira",
                                target="_blank",
                                className="social-media-icon-link"
                            ),
                        ],
                        className="social-media-icons",
                    )
                ],
                width=12,
            ),
        ],
        className="footer",
    ),
    
])




@app.callback(
    Output('pie-chart', 'figure'),
    [Input('year-dropdown', 'value')]
)

def update_pie_chart(selected_year):
    fig = px.pie(
        data_frame=data_df.loc[selected_year],
        names=data_df.columns,
        values=selected_year,
        title=f'Distribución de Gravedad por Tipo de Jornada - {selected_year}',
        labels={'index': 'Gravedad'}
    )
    return fig

@app.callback(
    Output('prediction-modal', 'children'),
    [Input('btn-prediction', 'n_clicks')]
)

def open_prediction_modal(n_clicks):
    if n_clicks:
        return dbc.Modal(
            [
                
                dbc.ModalBody(
                    [
                        
                        # Prediccion arbol de desición
                        html.H2("Realizar predicciones de gravedad en accidentes de transito dentro de Palmira"),
                        html.P('Aquí puedes realizar la predicción de la gravedad de un accidente de transito en el la ciudad de Palmira utilizando un modelo predictivo.'),
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
# Callback para realizar la predicción y mostrar el resultado
@app.callback(
    [Output('prediction-output', 'children'),
     Output('accidents-map', 'children')],
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('jornada-dropdown', 'value'),
     dash.dependencies.State('dia-semana-dropdown', 'value'),
     dash.dependencies.State('ubicacion-dropdown', 'value')]
)



def predict_callback(n_clicks, jornada, dia_semana, ubicacion):
    # Crear un DataFrame con los valores seleccionados por el usuario
    input_data = pd.DataFrame({
        'JORNADA': [jornada],
        'DIA_SEMANA': [dia_semana],
        'BARRIO-CORREGIMIENTO-VIA': [ubicacion]
    })

    # Aplicar codificación one-hot a los datos de entrada
    input_data_encoded = pd.get_dummies(input_data)

    # Asegurarte de que las columnas en input_data_encoded coincidan con las utilizadas en el modelo de regresión
    missing_cols = set(Xr_train.columns) - set(input_data_encoded.columns)
    for col in missing_cols:
        input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[Xr_train.columns]

    # Realizar las predicciones
    predicted_latitude = regression_model_lat.predict(input_data_encoded)
    predicted_longitude = regression_model_long.predict(input_data_encoded)
    predicted_gravity = classification_model.predict(input_data_encoded)

    # Crear un DataFrame con la ubicación predicha
    predicted_data = pd.DataFrame({
        'LAT': [predicted_latitude[0]],
        'LONG': [predicted_longitude[0]],
        'GRAVEDAD': [predicted_gravity[0]]
    })
    
    # Filtrar las ubicaciones reales que coinciden con los valores seleccionados por el usuario
    real_data = classification_data[(classification_data['JORNADA'] == jornada) &
                                    (classification_data['DIA_SEMANA'] == dia_semana) &
                                    (classification_data['BARRIO-CORREGIMIENTO-VIA'] == ubicacion)]
    
    # Concatenar los datos de predicción y ubicaciones reales
    map_data = pd.concat([predicted_data, real_data])
    
    # Eliminar filas con valores NaN en las columnas LAT y LONG
    map_data = map_data.dropna(subset=['LAT', 'LONG'])
    
    # Generar el mapa de Folium
    map_div = generate_folium_map(map_data)
    
    return f"Predicción - Latitud: {predicted_latitude[0]}, Longitud: {predicted_longitude[0]}, Gravedad: {predicted_gravity[0]}", map_div


if __name__ == '__main__':
    app.run_server(debug=True, port=80)
