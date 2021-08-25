import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import json

#fnames = ["res_figure10Mill.csv"]#"res_25avg.csv","res_35avg.csv"]#"resGIRG4.csv","GIRG-k-core.csv"]
#fnames = ["res_35_100000_ad.csv"]#["res_35avg.csv","res_35merge.csv",'res_25avg.csv']#"resGIRG4.csv","resMeta.csv"]
fnames = ["res_figure.csv"]

def plot3D(title,res,zlim=[-np.inf,np.inf],xlim=[-np.inf,np.inf],ylim=[-np.inf,np.inf],logz=1,logx=0, logy=1,color_scheme=0,dils=[25,20],statistic=0):
    data = res.loc[(res['title']==title)]
    rat = data.set_index(["tau","s","p"])["rat0"]
    pss = data.set_index(["tau"])["p"]
    pcs = data.set_index(["tau"])["p_c"]
    sss = data.set_index(["tau"])["s"]
    ns = data.set_index(["tau"])["n"]
    taus = data['tau'].unique()

    fig = go.Figure()
    
    for tau in taus:
        ps = pss[tau].unique()
        p_c = pcs[tau].values[0]
        ss = sss[tau].unique()
        n = ns[tau].values[0]
         
        if logx:
            imin=np.argmin(ps-p_c<0)
            ps = ps[(imin):]
            x = np.log(ps-p_c)/np.log(n)
            xlabel='log_n(p-p_c)'
        else:
            x = ps
            xlabel = 'p'
            
        if logy:
            y = np.log(ss)/np.log(n)
            ylabel='log_n(s)'
        else:
            y = ss
            ylabel='s'
                  
        z=[]
        vals={1:[],2:[],3:[],4:[],5:[]}
        for i,s in enumerate(ss):
            z.append([])
            for j,p in enumerate(ps):
                if isinstance(rat[tau][s][p],float):
                    zij = rat[tau][s][p]
                else:
                    zij_list = json.loads(rat[tau][s][p])
                    if statistic == 1:
                        zij = np.mean(zij_list[0])/np.mean(zij_list[1])
                    if statistic == 0:
                        zij = np.median(zij_list[0])/np.median(zij_list[1])
                if logz==1:
                    z[i].append(np.log(zij)/np.log(n))
                    zlabel="log_n(f(p,s))"
                elif logz==2:
                    if zij>=1:
                        z[i].append(np.log(zij)/np.log(n))
                    else:
                        z[i].append(min(0,-np.log(1-zij)/np.log(n)-1))
                    zlabel="log_n(f(p,s)) for f(p,s)>1, -log_n(1-f(p,s))-1 for f(p,s)<1"
                else:
                    z[i].append(zij)
                    zlabel="f(p,s)"
                if len(x)>j:
                    vals[f2(x[j]+0.2, y[i], tau, 0)[1]].append(z[i][-1])

                
        f_color=[]
        for i in range(len(y)):    
            f_color.append([])
            for j in range(len(x)):
                val=f2(x[j]+0.2, y[i], tau, 0)[1]
                if max(vals[val])-min(vals[val])==0:
                    f_color[i].append(val)     
                elif val==2:           
                    f_color[i].append(val-0.5*(z[i][j]-min(vals[val]))/(max(vals[val])-min(vals[val])) )
                else:
                    f_color[i].append(val+0.5*(z[i][j]-min(vals[val]))/(max(vals[val])-min(vals[val])) )

        z=np.array(z)

        x = np.maximum(np.minimum(x,xlim[1]),xlim[0])
        y = np.maximum(np.minimum(y,ylim[1]),ylim[0])
        z = np.maximum(np.minimum(z,zlim[1]),zlim[0])
        
        if color_scheme==1:
            #fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50))
            fig.add_trace(go.Surface(x=x, y=y, z=z, visible=False,
                                     cmin=1, cmax=6,surfacecolor=f_color, colorscale=["#FFFF00","#FFFFFF","#70AA00","#FFFFFF","#00FFFF","#FFFFFF","#F7931E","#FFFFFF","#FF0000","#FFFFFF","#FFFFFF"] ,
                                     name="tau = " + str(tau)))
        elif color_scheme==0:
            plasma=['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921', '#f0f921', '#f0f921']
            colorS=[]
            from colour import Color

            for i in range(len(plasma)-1):
                if i in [4,5]:
                    colorS.append(plasma[i])
                else:
                    if i<4:
                        dil=int(dils[0])
                    else:
                        dil=int(dils[1])
                    
                    fineColors = Color(plasma[i]).range_to(Color(plasma[i+1]),dil)
                    for color in fineColors:
                        colorS.append(color.hex)
            #print(colorS)
            #fig.add_trace(go.Mesh3d(x=x, y=y, z=z,color='lightpink', opacity=0.50))
            fig.add_trace(go.Surface(x=x, y=y, z=z,visible=False, cmin=-1, cmax=1,
                                     name="tau = " + str(tau), colorscale=colorS))
            
        if len(taus)==1:
            zlabel+=", tau={}".format(taus[0])
        else:
            zlabel+=", tau=(see slider)"            
        fig.update_layout(title_text=zlabel)  #for x>0, 1-log_n(1-f(p,s)) for x<0")
    
        fig.update_layout(scene = dict(
                            xaxis_title=xlabel,
                            yaxis_title=ylabel,
                            zaxis_title=zlabel[:min(len(zlabel),13)],
                            aspectmode='cube'))
    
        
    fig.data[0].visible = True
    
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            label=str(taus[i]),
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "tau: "},
        pad={"t": 50},
        steps=steps,
    )]
    
    fig.update_layout(
        sliders=sliders,
        width=900,
        height=800,
    )

    #fig.show()
    return fig

def f2(x,y,tau,diffnorm=False):
    taum1=tau-1
    taum2=tau-2
    taum3=np.abs(tau-3) 
    if tau>3:
        slope=taum2/taum3
        slope2=1
    else:
        slope=1/taum3
        slope2=0
        
    if y>=1+taum2/taum3*x: #A5
        if tau>3:
            return 1/(taum1*taum2)*(1-y), 5
        else:
            return (1-1/taum1)*(1-y), 5
    elif (y>=1+taum1/taum3*x and y <= -1/taum3*x) or (y>=1+1/taum3*x): #A3
        return slope2*x+(1-1/taum1)*(1-y), 3
#        return -1/taum1-2/taum3*x+(1/taum1-1)*y
    elif y <= -1/taum3*x:
        return 1+slope*x-y, 1 #A1
#        return -1/taum3*x-y
    elif y>=1+taum1/taum3*x:
        return -1/taum3*x-1/taum1*(1-y), 4 #A4
    else:
        if diffnorm:
            return slope*x-y, 2 #A2
        else:
            return 0, 2 #A2

def plot3DTheoretic(diffnorm=False,sh_0=50,sh_1=50,color_scheme=0):
    fig = go.Figure()
    taus=np.linspace(2.1,4,11)
    #taus=np.linspace(3.1,4,11)
    for tau in taus:
        x, y = np.linspace(0,-np.abs(tau-3)/(tau-1), sh_0), np.linspace(1,0, sh_1)
        X, Y = np.meshgrid(x, y)
        #Z=[]
        Z2=[]
        vals={1:[],2:[],3:[],4:[],5:[]}
        for i in range(len(X)):
            #Z.append([])
            Z2.append([])
            for j in range(len(X[0])):
                #Z[i].append(f(X[i][j], Y[i][j], tau))
                Z2[i].append(f2(X[i][j], Y[i][j], tau, diffnorm)[0])
                vals[f2(X[i][j], Y[i][j], tau, diffnorm)[1]].append(Z2[i][-1])
        f_color=[]
        for i in range(len(X)):    
            f_color.append([])
            for j in range(len(X[0])):
                val=f2(X[i][j], Y[i][j], tau, diffnorm)[1]
                if max(vals[val])-min(vals[val])==0:
                    f_color[i].append(val)     
                elif val==2:           
                    f_color[i].append(val-0.5*(Z2[i][j]-min(vals[val]))/(max(vals[val])-min(vals[val])) )
                else:
                    f_color[i].append(val+0.5*(Z2[i][j]-min(vals[val]))/(max(vals[val])-min(vals[val])) )
        #print(f_color)

    #colorscale=['#006400', '#007500', '#008600', '#009700', '#00a800', '#00b900', '#00ca00', '#00db00', '#00ec00', '#0d0887','#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']
        #print("color_scheme",color_scheme)
        if color_scheme==1:
            fig.add_trace(go.Surface(x=x, y=y, z=Z2,visible=False, cmin=1, cmax=6,surfacecolor=f_color, colorscale=["#FFFF00","#FFFFFF","#70AA00","#FFFFFF","#00FFFF","#FFFFFF","#F7931E","#FFFFFF","#FF0000","#FFFFFF","#FFFFFF"] , name="tau = " + str(tau)))
        elif color_scheme==0:
            fig.add_trace(go.Surface(x=x, y=y, z=Z2,visible=False, cmin=-1, cmax=1, name="tau = " + str(tau)))
            
        if diffnorm:
            fig.update_layout(title_text="Limit of log_n(f(p,s)) for f(p,s)>1, -log_n(1-f(p,s))-1 for f(p,s)<1, tau=(see slider)")
        else:
            fig.update_layout(title_text="Limit of log_n(f(p,s)), tau=(see slider)")
            
    
        fig.update_layout(scene = dict(
                            xaxis_title='x',
                            yaxis_title='y',
                            zaxis_title='log_n(f(p,s))',
                            aspectmode='cube'))
    
        
    fig.data[0].visible = True
    
    
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],  # layout attribute
            label=str(taus[i]),
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "tau: "},
        pad={"t": 50},
        steps=steps,
    )]
    
    fig.update_layout(
        sliders=sliders,
        width=900,
        height=800,
    )
    return fig

def create_app(res):
    titles=["Theoretic config"]+list(res['title'].unique())
    app = dash.Dash()
    app.layout = html.Div([
        html.Div(["Data: ",
                dcc.Dropdown(
                    id='data_type',
                    options= [{'label': l, 'value': i} for i,l in enumerate(titles)],
                    value=0,
                )],style={'width': '25%', 'display': 'inline-block'}),
        dcc.Graph(id='graph'),
        html.Div(["x axis:",
                dcc.RadioItems(
                    id='xaxis-type',
                )],style={'width': '35%', 'display': 'inline-block'}),
        html.Div(["xlim: ",
              dcc.Input(id='xlim', value='(-inf,inf)', type='text')],style={'width': '65%', 'display': 'inline-block'}),
        html.Div(["y axis:",
                dcc.RadioItems(
                    id='yaxis-type',
                )],style={'width': '35%', 'display': 'inline-block'}),
        html.Div(["ylim: ",
              dcc.Input(id='ylim', value='(-inf,inf)', type='text')],style={'width': '65%', 'display': 'inline-block'}),
        html.Div(["z axis:",
                dcc.RadioItems(
                    id='zaxis-type',
                )],style={'width': '35%', 'display': 'inline-block'}),
        html.Div(["zlim: ",
              dcc.Input(id='zlim', value='(-inf,inf)', type='text')],style={'width': '65%', 'display': 'inline-block'}),
        #html.Div(["color:",
        #        dcc.RadioItems(
        #            id='color-scheme',
        #        )],style={'width': '100%', 'display': 'inline-block'}),
        #html.Div(["color dilutions: ",
        #      dcc.Input(id='dils', value='(20,20)', type='text')],style={'width': '65%', 'display': 'inline-block'}),
        html.Div(["statistic:",
                dcc.RadioItems(
                    id='statistic',
                )],style={'width': '35%', 'display': 'inline-block'}),
    ])
    

    @app.callback(
        Output(component_id='xaxis-type', component_property='options'),
        Output(component_id='yaxis-type', component_property='options'),
        Output(component_id='zaxis-type', component_property='options'),
        Output(component_id='statistic', component_property='options'),
    #    Output(component_id='color-scheme', component_property='options'),
        Output(component_id='xaxis-type', component_property='value'),
        Output(component_id='yaxis-type', component_property='value'),
        Output(component_id='zaxis-type', component_property='value'),
    #    Output(component_id='color-scheme', component_property='value'),
        Output(component_id='statistic', component_property='value'),
        Input(component_id='data_type', component_property='value'),
    )
    def changeData(data_type):
        statistic = 0
    #    color_scheme=[{'label': 'plasma', 'value': 0},{'label': 'theoretic', 'value': 1}]
        if data_type==0:
            xoptions = [{'label': 'log_n(p-p_c)', 'value': 1}]
            yoptions = [{'label': 'log_n(s)', 'value': 1}]
            zoptions = [{'label': 'log_n(f)', 'value': 1}, {'label': 'log_n(f) for f>1, 1-log_n(1-f) for f<1', 'value': 2} ]
            xvalue = 1
            yvalue = 1
            zvalue = 1  
            soptions=[{'label': l, 'value': i} for i,l in enumerate([])]

    #        cvalue = 1      
        else:
            xoptions=[{'label': l, 'value': i} for i,l in enumerate(['p', 'log_n(p-p_c)'])]
            yoptions=[{'label': l, 'value': i} for i,l in enumerate(['s', 'log_n(s)'])]
            zoptions=[{'label': l, 'value': i} for i,l in enumerate(['f', 'log_n(f)', 'log_n(f) for f>1, 1-log_n(1-f) for f<1'])]
            xvalue = 1
            yvalue = 1
            zvalue = 1
            soptions=[{'label': l, 'value': i} for i,l in enumerate(['median', 'mean'])]

    #        cvalue = 0    
    #    return (xoptions, yoptions, zoptions,soptions,color_scheme,xvalue,yvalue,zvalue,cvalue,statistic)
        return (xoptions, yoptions, zoptions,soptions,xvalue,yvalue,zvalue,statistic)


    def tofloat(st):
        try:
            st1,st2 = st.split(",")
            try:
                fl1=float(st1[1:])
            except:
                fl1 = -np.inf
            try:
                fl2=float(st2[:-1])
            except:
                fl2 = np.inf
        except:
            fl1 = -np.inf
            fl2 = np.inf
        if fl1==np.inf:
            fl1=-np.inf
        if fl2==-np.inf:
            fl2=np.inf
        return [fl1,fl2]

    @app.callback(
        Output(component_id='graph', component_property='figure'),
        Input(component_id='data_type', component_property='value'),
        Input(component_id='zlim', component_property='value'),
        Input(component_id='xlim', component_property='value'),
        Input(component_id='ylim', component_property='value'),
        Input(component_id='xaxis-type', component_property='value'),
        Input(component_id='yaxis-type', component_property='value'),
        Input(component_id='zaxis-type', component_property='value'),
        #Input(component_id='color-scheme', component_property='value'),    
        #Input(component_id='dils', component_property='value'),
        Input(component_id='statistic', component_property='value'),
    )
    #def updateplot(data_type,zlim,xlim,ylim,logx,logy,logz,color_scheme,dils,statistic):
    #def updateplot(data_type,zlim,xlim,ylim,logx,logy,logz,color_scheme,statistic):
    def updateplot(data_type,zlim,xlim,ylim,logx,logy,logz,statistic):
        if data_type==0:
            return plot3DTheoretic(diffnorm=(logz==2),color_scheme=1)
        else:
            #return plot3D(title=titles[data_type],
            #    zlim=tofloat(zlim),xlim=tofloat(xlim),ylim=tofloat(ylim),
            #    logx=logx,logy=logy,logz=logz,color_scheme=color_scheme, dils=tofloat(dils), statistic=statistic)
            return plot3D(title=titles[data_type],res=res,
                          zlim=tofloat(zlim),xlim=tofloat(xlim),ylim=tofloat(ylim),
                          logx=logx,logy=logy,logz=logz,color_scheme=0, statistic=statistic)
        
    return app

if __name__ == '__main__':
    res=pd.DataFrame()
    for fname in fnames:
        res=res.append(pd.read_csv(fname))
    
    app = create_app(res)
    server = app.server
    app.run_server(debug=True,port=8086)
    
