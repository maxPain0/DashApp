# DashApp
import numpy as np
import struct
import pandas as pd
import matplotlib.pyplot as plt
import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import json
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff

# Function to add frame number to df
# Matrix has always 2907 singel points
def add_frame_nr(df,i,k,amount_frames):
    #input:
    #df = dataframe
    #i,k = counter
    #amount_frames = how many frames are in the df

    while i < amount_frames:
        if i == 0:
            iloc_1 = 0
        if i > 0:
            iloc_1 = k - 2907
        iloc_2 = k
        df.loc[iloc_1:iloc_2,'frames'] = int(i)
        k += 2907
        i += 1

    #return the df with column 'frames' = framenr
    return df

def read_pc2(path,number = 0):
    #input:
    #path = path where the pc2 file is located
    #number = number of the frame starting from 0, also 0 if nothing is changed

    with open(path, 'rb') as f:
        head_fmt = '<12siiffi'
        data_fmt = '<fff'
        head_unpack = struct.Struct(head_fmt).unpack_from
        data_unpack = struct.Struct(data_fmt).unpack_from
        data_size = struct.calcsize(data_fmt)
        headerStr = f.read(struct.calcsize(head_fmt))
        head = head_unpack(headerStr)
        nverts, nframes = head[2], head[5]
        data = []
        for i in range(nverts*nframes):
            data_line = f.read(data_size)
            if len(data_line) != data_size:
                return None
            data.append(list(data_unpack(data_line)))
        data = np.array(data).reshape([nframes, nverts, 3])
    arr_reshaped = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
    df = pd.DataFrame(arr_reshaped)
    df.columns = ['x_val2','y_val2','z_val2']
    df = add_frame_nr(df,0,2907,df.shape[0]/2907)
    df['frames'] = df['frames'].astype(int)
    df = df[(df["frames"]==number)]
    df.index = np.arange(start = 0, stop = len(df) , step = 1)
    df.to_csv("Dataframe_to_csv" +str(number) + '.csv', sep=';' )

    # return: dataframe with manipulated df
    return df

### Input starts here:
#Example:
# df = read_pc2(r'C:\files\KLA0057Animation_stabilised.pc2', 1)
#Frames start with 0

Frame_one = 0
Frame_two = 71
df = read_pc2(r'C:\files\KLA0057Animation_stabilised.pc2', Frame_one)
df2 = read_pc2(r'C:\files\KLA0057Animation_stabilised.pc2', Frame_two)

array_1 = df.to_numpy()
array_2 = df2.to_numpy()

# def test_function():
#     i =0
#     while i < 71:
#         df = read_pc2(r'C:\files\KLA0057Animation_stabilised.pc2', 0)
#         df2 = read_pc2(r'C:\files\KLA0057Animation_stabilised.pc2', i)
#         array_1 = df.to_numpy()
#         array_2 = df2.to_numpy()
#         val_directed_hausdorff = directed_hausdorff(array_1, array_2)[0]
#         if i == 0:
#             old_directed_hausdorff = val_directed_hausdorff
#         elif val_directed_hausdorff > old_directed_hausdorff:
#             old_directed_hausdorff = val_directed_hausdorff
#             biggest_dist_frame = i
#         i = i +1
#     return biggest_dist_frame 
# print('Test frame fkt' + str(test_function()))
    


def time_calculator(Frame_one,Frame_two):  
    return (Frame_two - Frame_one) /60

time_in_seconds_frame_1_to_2 = time_calculator(Frame_one,Frame_two)

############################################################################
#
#
## Here starts the Dash App
#
#
############################################################################

# load externa stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

# Creation of Figure
# Crate a Scatter3d figure out of the first dataframe = df
import plotly.graph_objects as go
def create_figure(skip_points=[]):
    dfs = df.drop(skip_points)
    fig = go.Figure(data =[go.Scatter3d(x = dfs['x_val2'],
                                   y = dfs['y_val2'],
                                   z = dfs['z_val2'],
                                   mode ='markers',
                                   marker = dict(
                                     color="blue",
                                     size = 5,
                                     opacity = 0.6,
                                    line=dict(
                                        color='black',
                                        width=1),
                                    

                                   )
)])
    fig.update_layout(uirevision='wrong')
    return fig
f = create_figure()


# Define HTML Layout
app.layout = html.Div(
                    [html.Button('Delete', id='delete'),
                    html.Button('Clear Selection', id='clear'),
                    html.Button('Procrustes analysis', id='procrustes_analyses'),
                    dcc.Graph(id = '3d_scat', figure=f,style={'width': '50%','height':'1080px','padding-left':'25%', 'padding-right':'25%'}),
                    html.Div('selected:'),
                    html.Div(id='selected_points'), #, style={'display': 'none'})),
                    html.Div('deleted:'),
                    html.Div(id='deleted_points'),
                    html.Div('Procrustes:'),
                    html.Div(id='body-div')
])

# Function to delete Selected Points:
@app.callback(
            Output('deleted_points', 'children'),
            Input('delete', 'n_clicks'),
            State('selected_points', 'children'),
            State('deleted_points', 'children')
 )

#function that deletes the selected datapoints (not needed in the application I think)
def delete_points(n_clicks, selected_points, delete_points):
    #print('n_clicks:',n_clicks)
    if selected_points:
        selected_points = json.loads(selected_points)
    else:
        selected_points = []

    if delete_points:
        deleted_points = json.loads(delete_points)
    else:
        deleted_points = []
    ns = [p['pointNumber'] for p in selected_points]
    new_indices = [df.index[n] for n in ns if df.index[n] not in deleted_points]
    print('new',new_indices)
    deleted_points.extend(new_indices)
    return json.dumps(deleted_points)

# Funciton to select the Datapoint into a List
    
@app.callback(
        Output('selected_points', 'children'),
        Input('3d_scat', 'clickData'),
        Input('deleted_points', 'children'),
        Input('clear', 'n_clicks'),
        State('selected_points', 'children'))



def select_point(clickData, deleted_points, clear_clicked, selected_points):
    ctx = dash.callback_context
    ids = [c['prop_id'] for c in ctx.triggered]

    if selected_points:
        results = json.loads(selected_points)
        #print(selected_points)
        #my_array = np.asarray(results)
        #print(my_array)
    else:
        results = []
    if '3d_scat.clickData' in ids:
        if clickData:
            for p in clickData['points']:
                if p not in results:
                    results.append(p)
    if 'deleted_points.children' in ids or  'clear.n_clicks' in ids:
        results = []
    results = json.dumps(results)
    return results


#Callback to create Figure:
@app.callback(
            Output('3d_scat', 'figure'),
            Input('selected_points', 'children'),
            Input('deleted_points', 'children'),
            State('deleted_points', 'children'))

def chart_3d( selected_points, deleted_points_input, deleted_points_state):
    global f
    deleted_points = json.loads(deleted_points_state) if deleted_points_state else []
    f = create_figure(deleted_points)

    selected_points = json.loads(selected_points) if selected_points else []
    if selected_points:
        f.add_trace(
            go.Scatter3d(
                mode='markers',
                x=[p['x'] for p in selected_points],
                y=[p['y'] for p in selected_points],
                z=[p['z'] for p in selected_points],
                marker=dict(
                    color='white',
                    size=6,
                    line=dict(
                        color='red',
                        width=3
                    )
                ),
                showlegend=False
            )
        )
    return f

# Function to delete Selected Points:
@app.callback(
            Output('body-div', 'children'),
            Input('procrustes_analyses', 'n_clicks'),
            State('selected_points', 'children'),
            State('body-div', 'children')
 )

#Function to convert the selected datapoints into an numpy array. 

def procrustes_points(n_clicks, selected_points, delete_points):
    print('n_clicks:',n_clicks)
    y = json.loads(selected_points)
    k=0

    for i in y:
        data = {'x':i["x"], 'y':i["y"], 'z':i["z"]}
        data_point = {'Datapoint':i['pointNumber']}
        print(data_point)
        if k == 0:
            df_procrustes_1 = pd.DataFrame(data, index=[k])
            df_procrustes_2 = df2.iloc[[data_point.get('Datapoint')]]
            k = k+1
        else:
            df_procrustes_2 = df_procrustes_2.append([df2.iloc[data_point.get('Datapoint')]],ignore_index=True)
            df_procrustes_1 = df_procrustes_1.append(data,ignore_index=True)            
            k = k+1

    df_procrustes_2.drop('frames', axis=1, inplace=True)
    A = df_procrustes_1.to_numpy()
    B = df_procrustes_2.to_numpy()

    ########################################################################
    #
    #
    #Procrustes Start
    #
    #
    #########################################################################
   
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html
    mtx1, mtx2, disparity = procrustes(A, B)
    print('rounded disparity '+ str(round(disparity)))
    print('disparity '+ str(disparity))
    print('mtx1 '+ str(mtx1))
    print('mtx2 '+ str(mtx2))

    # This is just in the code due to my interest
    print('Hausdorf distance ' + str(directed_hausdorff(array_1, array_2)[0]))
    

    ########################################################################
    #
    #
    #Procrustes End
    #
    #
    #########################################################################



    ########################################################################
    #
    #
    #Disdance Start
    #
    #
    #########################################################################
    def point_distance_calculator(selcted_points_in_plot):
        p1 = np.array([A[selcted_points_in_plot,0],A[selcted_points_in_plot,1],A[selcted_points_in_plot,2]])
        p2 = np.array([B[selcted_points_in_plot,0],B[selcted_points_in_plot,1],B[selcted_points_in_plot,2]])
        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        return dist

    print ('Distance betweeen two Points 0:' + str(point_distance_calculator(0)))
    print ('Distance betweeen two Points 1:' + str(point_distance_calculator(1)))
    print ('Distance betweeen two Points 2:' + str(point_distance_calculator(2)))
    print ('Speed betweeen two Points 0:' + str([point_distance_calculator(0) /  time_in_seconds_frame_1_to_2 ]))
    print ('Speed betweeen two Points 0:' + str([point_distance_calculator(1) /  time_in_seconds_frame_1_to_2 ]))
    print ('Speed betweeen two Points 0:' + str([point_distance_calculator(2) /  time_in_seconds_frame_1_to_2 ]))
    print ('It took from minimal to maximal movement the following ammount of seconds ' + str(time_in_seconds_frame_1_to_2))

    ########################################################################
    #
    #
    #Disdance End
    #
    #
    #########################################################################

    
    return "Disparity: "+ str(disparity) + ' /     A: ' + str(A) +   ' /     B: ' + str(B) + ' / Standardized    A: ' + str(mtx1) + ' / Standardized    B: ' + str(mtx2) 






if __name__ == '__main__':
    app.run_server(debug=False, port=8044)
