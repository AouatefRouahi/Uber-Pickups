# ***************************************************************************************************************
# Import Librairies
# ***************************************************************************************************************

# generic libs
import pandas as pd
import numpy as np
import calendar
from datetime import datetime

# ML libs
from sklearn.cluster import DBSCAN

# geometry libs
from shapely.geometry import MultiPoint
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geoapiExercises")

# plot libs
import plotly.express as px

# global params
file_path = "data/pre_aug14.csv"

# Application libs
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# ***************************************************************************************************************
# Useful functions
# ***************************************************************************************************************

# clustering using dbscan
def get_clusters(max_distance,min_pickups,pickups):
    
    ## get pickups coordinates from available data
    coords = pickups[['Lat', 'Lon']].to_numpy()
        
    # convert distances from kilometers to radians, as measured along a great circle on a sphere 
    # with a radius of 6371 km, the mean radius of the Earth.
    kms_per_radian = 6371.0088
    
    # convert coordinates and epsilon to radians
    epsilon = max_distance / kms_per_radian
    coords_rad = np.radians(coords)
    
    ##  DBSCAN clustering
    db = DBSCAN(eps=epsilon, min_samples=min_pickups, metric='haversine')
    db.fit(coords_rad)
    
    ## count the clusters
    cluster_labels = db.labels_
    n_clusters = len(set(cluster_labels))
    print('Number of clusters: {}'.format(n_clusters))
    
    # annotate the pickups with their equivalent clusters
    annotated_pickups = np.insert(coords, 2, cluster_labels, axis = 1)
   
    return annotated_pickups, n_clusters

# get centermost_point of each cluster
def get_centermost_point(cluster):
    # centroid : tuple (latitude, longitude)
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    # centermost : one point in the cluster that gives the shortest distance to the centroid in meters(m)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)

    return tuple(centermost_point)

def get_centermost_points(annotated_pickups, n_clusters):
    # return centermost coordinates (lat, lon) and the clusters' sizes
    # annotated_pickups [lat, lon, cluster] : np array
    
    # initialize lists for centermost_point and size
    lats = []
    lons = []
    sizes = []
    
    # grouping clusters (without considering the noise cluster)
    for c in range(n_clusters-1):
        # get all pickups in the cluster "c"
        mask = (annotated_pickups[:, 2] == c)
        cluster = annotated_pickups[mask, :][:,0:2]
        
        # filter empty clusters
        if cluster.any():
            centermost =  get_centermost_point(cluster)
            lats.append(centermost[0]) 
            lons.append(centermost[1])
            sizes.append(len(cluster))
            
    # make the computed data in a df     
    centermosts = [lats,lons,sizes]
    df_centermosts =  pd.DataFrame({'lat':lats, 'lon':lons, 'size':sizes})
    
    return df_centermosts

# get location information of the centermost_point of each cluster
def get_location_info(latitude, longitude):
    location = str( geolocator.reverse(str(latitude)+","+str(longitude)))
    return location

# ***************************************************************************************************************
# Get recommendations
# ***************************************************************************************************************

#1. load data
pickups = pd.read_csv(file_path)

# Dash part
app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children="UBER PICKUPS!"),
    html.P(children="Select the day of week:"),
    dcc.RadioItems(options=[
       {'label': 'Monday', 'value': '0'},
       {'label': 'Tuesday', 'value': '1'},
       {'label': 'Wednesday', 'value': '2'},
       {'label': 'Thursday', 'value': '3'},
       {'label': 'Friday', 'value': '4'},
       {'label': 'Saturday', 'value': '5'},
       {'label': 'Sunday', 'value': '6'},
       ],
       value='0',
       id='day-of-week'),
    html.P(children="Select the time range:"),
    dcc.RangeSlider(0, 23, 1, value=[5, 15], id='time-range'),
    html.P(children="Enter the maximum distance between two pickups (in km):"),
    dcc.Input(id='max_dis', type='number', value=0.05),
    html.P(children="Enter the minimum number of pickups by zone:"),
    dcc.Input(id="min_pick", type="number", value=25),   
    html.H3(children="UBER MAP"),
    dcc.Graph(id="my-map"),  
])


@app.callback(
    Output("my-map", "figure"),
    Input("day-of-week", "value"),
    [Input('time-range', "value")],
    Input("max_dis", "value"),
    Input("min_pick", "value"),
)
def generate_map(weekday, time_range, max_dis, min_pick):
    #1. get the driver query and filter the dataset
    weekday_order = int(weekday)
    weekday_name = calendar.day_name[weekday_order]
    hour_1 = datetime.strptime(str(time_range[0]), "%H").strftime("%I %p")
    hour_2 = datetime.strptime(str(time_range[1]), "%H").strftime("%I %p")

    print(f'weekday:{weekday_name} \nhour :{time_range} \nmax_distance:{float(max_dis)}\nmin_pickups:{min_pick}')
    mask = (pickups['weekday']== int(weekday)) & (pickups['hour'] >= time_range[0]) & (pickups['hour'] <= time_range[1])
    df = pickups[mask].reset_index()

    # maximum distance between two cluster members in kilometers
    max_distance = float(max_dis)

    # minimum number of cluster members
    min_pickups = int(min_pick)
    
    #2. get hot zones
    annotated_pickups, n_clusters= get_clusters(max_distance, min_pickups, df)

    # get zones centermost locations & the number of pickups in each zone (size)
    df_centermosts = get_centermost_points(annotated_pickups, n_clusters)
    df_centermosts['location_info'] = df_centermosts.apply(lambda row: get_location_info(row.lat, row.lon), axis = 1)

    #3. Visualize the hot-zones
    title = f'Hot Spots for {weekday_name} between {hour_1} and {hour_2}'
    fig = px.scatter_mapbox(
        df_centermosts, 
        lat= "lat", 
        lon = "lon", 
        size = 'size',
        color = 'size',
        text = 'location_info',
        mapbox_style = "carto-positron",
        zoom = 10,
        title = title,
        width = 1500, height = 600  
    )
    return fig

# ***************************************************************************************************************
# Run the application
# ***************************************************************************************************************

if __name__ == '__main__':
    app.run_server(debug=False)