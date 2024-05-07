import streamlit as st
import plotly.express as px
from plotly_football_pitch import make_pitch_figure, PitchDimensions, SingleColourBackground
import pandas as pd 
import numpy as np
import plotly_football_pitch as pfp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
st.set_page_config(layout='wide')
symbol_map = {
    'left': 'circle',
    'right': 'circle',
    'head': 'square'
}
color_discrete_map = {'Goal': 'green', 'No Goal': 'red'}

stype = ['All shots','From Corner','Counter Attack','From Cross','Free Kick']
@st.cache_data
def load_data():
    data = pd.read_csv('shotsfor.csv')
    data['goal'] = np.where(data['shotOutcome']=='goal',True,False)
    data['goal_label'] = data['goal'].map({True: 'Goal', False: 'No Goal'})
    data['marker_symbol'] = data['footName'].map(symbol_map)
    data = data.sort_values('goal_label',ascending=True)
    data = data.reset_index(drop='index')
    return data
def load_opp_data():
    data = pd.read_csv('shotsag.csv')
    data['goal'] = np.where(data['shotOutcome']=='goal',True,False)
    data['goal_label'] = data['goal'].map({True: 'Goal', False: 'No Goal'})
    data['marker_symbol'] = data['footName'].map(symbol_map)
    data = data.sort_values('goal_label',ascending=True)
    data = data.reset_index(drop='index')
    return data


def create_pitch_plot_all(data):
    # Define pitch dimensions and create a pitch figure
    dimensions = PitchDimensions(100, 101)
    fig = pfp.make_pitch_figure(
        dimensions,
        figure_height_pixels=800,
        figure_width_pixels=700,
        pitch_background=pfp.SingleColourBackground("white"),
        orientation=pfp.PitchOrientation.VERTICAL
    )

    # Create the scatter plot
    scatter = px.scatter(data, x=data['y'], y=data['x'], log_x=True,
                         hover_name="player", hover_data=['shotOutcome', "xg", "possession","footName"],
                         color='goal_label',color_discrete_map=color_discrete_map)
    for trace in scatter.data:
        fig.add_trace(trace)


    # Update traces to adjust marker size and style
    fig.update_traces(marker_size=12, 
                      marker_opacity=0.7, 
                      marker_line_color='black', 
                      marker_line_width=1) 

    return fig

def create_pitch_plot_player(data):
    # Define pitch dimensions and create a pitch figure
    dimensions = PitchDimensions(100, 101)
    fig = pfp.make_pitch_figure(
        dimensions,
        figure_height_pixels=800,
        figure_width_pixels=700,
        pitch_background=pfp.SingleColourBackground("white"),
        orientation=pfp.PitchOrientation.VERTICAL
    )

    # Create the scatter plot
    scatter = px.scatter(data, x=data['y'], y=data['x'], log_x=True,
                         hover_name="player", hover_data=['shotOutcome', "xg", "possession"],
                         color='goal_label', color_discrete_map={'Goal': 'green', 'No Goal': 'red'})

    # Add scatter plot data to the pitch figure
    for trace in scatter.data:
        fig.add_trace(trace)

    # Update traces to adjust marker size and style
    fig.update_traces(marker_size=12, 
                      marker_opacity=0.7, 
                      marker_line_color='black', 
                      marker_line_width=1)

    return fig

fora = st.radio(label='Shots for or Against',options=['For','Against'])
if fora == 'For':
    data = load_data()
    p = data.player.unique().tolist()
    p.insert(0,'All Players')
    player_name = st.selectbox('Select Player: ',options=p,placeholder='All Players')
    if player_name == 'All Players':
        shotstype = st.selectbox('Select shot type: ',options = list(stype))
        if shotstype == 'All shots':
            col1, col2 = st.columns(2)
            with col1:
                    data = data
                    mins = data.groupby('minute').size().reset_index()
                    mins.columns = ['minute','shots']
                    goals = data[data['goal']==True].reset_index()
                    goals = goals.groupby('minute').size().reset_index()
                    goals.columns = ['minute','goals']
                    fig = create_pitch_plot_all(data)
                    fig.update_layout(title='Shot distribution')
                    #final_fig.update_layout(width=4000)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    mins['time_interval'] = pd.cut(mins['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = mins.groupby('time_interval')['shots'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['shots'],
                        marker_color='blue')])
                    fig.update_layout(
                            title='Distribution of Shots by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Shots',
                            template='plotly_white')
                    st.plotly_chart(fig)
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    goals['time_interval'] = pd.cut(goals['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = goals.groupby('time_interval')['goals'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['goals'],
                        marker_color='red')])
                    fig.update_layout(
                            title='Distribution of Goals by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Goals',
                            template='plotly_white')
                    st.plotly_chart(fig)
        if shotstype == 'From Corner':
            col1,col2 = st.columns(2)
            with col1:
                    data = data[data['possession'].str.contains('Corner')].reset_index(drop='index')
                    mins = data.groupby('minute').size().reset_index()
                    mins.columns = ['minute','shots']
                    goals = data[data['goal']==True].reset_index()
                    goals = goals.groupby('minute').size().reset_index()
                    goals.columns = ['minute','goals']
                    fig = create_pitch_plot_all(data)
                    fig.update_layout(title='Shot distribution')
                    #final_fig.update_layout(width=4000)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    mins['time_interval'] = pd.cut(mins['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = mins.groupby('time_interval')['shots'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['shots'],
                        marker_color='blue')])
                    fig.update_layout(
                            title='Distribution of Shots by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Shots',
                            template='plotly_white')
                    st.plotly_chart(fig)
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    goals['time_interval'] = pd.cut(goals['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = goals.groupby('time_interval')['goals'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['goals'],
                        marker_color='red')])
                    fig.update_layout(
                            title='Distribution of Goals by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Goals',
                            template='plotly_white')
                    st.plotly_chart(fig)
        if shotstype == 'Counter Attack':
            col1,col2 = st.columns(2)
            with col1:
                    data = data[data['possession'].str.contains('Counterattack')].reset_index(drop='index')
                    mins = data.groupby('minute').size().reset_index()
                    mins.columns = ['minute','shots']
                    goals = data[data['goal']==True].reset_index()
                    goals = goals.groupby('minute').size().reset_index()
                    goals.columns = ['minute','goals']
                    fig = create_pitch_plot_all(data)
                    fig.update_layout(title='Shot distribution')
                    #final_fig.update_layout(width=4000)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    mins['time_interval'] = pd.cut(mins['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = mins.groupby('time_interval')['shots'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['shots'],
                        marker_color='blue')])
                    fig.update_layout(
                            title='Distribution of Shots by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Shots',
                            template='plotly_white')
                    st.plotly_chart(fig)
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    goals['time_interval'] = pd.cut(goals['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = goals.groupby('time_interval')['goals'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['goals'],
                        marker_color='red')])
                    fig.update_layout(
                            title='Distribution of Goals by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Goals',
                            template='plotly_white')
                    st.plotly_chart(fig)
        if shotstype == 'Free Kick':
            col1,col2 = st.columns(2)
            with col1:
                    data = data[data['possession'].str.contains('Free kick')].reset_index(drop='index')
                    mins = data.groupby('minute').size().reset_index()
                    mins.columns = ['minute','shots']
                    goals = data[data['goal']==True].reset_index()
                    goals = goals.groupby('minute').size().reset_index()
                    goals.columns = ['minute','goals']
                    fig = create_pitch_plot_all(data)
                    fig.update_layout(title='Shot distribution')
                    #final_fig.update_layout(width=4000)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    mins['time_interval'] = pd.cut(mins['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = mins.groupby('time_interval')['shots'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['shots'],
                        marker_color='blue')])
                    fig.update_layout(
                            title='Distribution of Shots by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Shots',
                            template='plotly_white')
                    st.plotly_chart(fig)
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    goals['time_interval'] = pd.cut(goals['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = goals.groupby('time_interval')['goals'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['goals'],
                        marker_color='red')])
                    fig.update_layout(
                            title='Distribution of Goals by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Goals',
                            template='plotly_white')
                    st.plotly_chart(fig)
        if shotstype == 'From Cross':
            col1,col2 = st.columns(2)
            with col1:
                    data = data[data['types'].str.contains('from_cross')].reset_index(drop='index')
                    mins = data.groupby('minute').size().reset_index()
                    mins.columns = ['minute','shots']
                    goals = data[data['goal']==True].reset_index()
                    goals = goals.groupby('minute').size().reset_index()
                    goals.columns = ['minute','goals']
                    fig = create_pitch_plot_all(data)
                    fig.update_layout(title='Shot distribution')
                    #final_fig.update_layout(width=4000)
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    mins['time_interval'] = pd.cut(mins['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = mins.groupby('time_interval')['shots'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['shots'],
                        marker_color='blue')])
                    fig.update_layout(
                            title='Distribution of Shots by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Shots',
                            template='plotly_white')
                    st.plotly_chart(fig)
                    bins = [0, 15, 30, 45, 60, 75, 90]
                    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                    goals['time_interval'] = pd.cut(goals['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                    interval_data = goals.groupby('time_interval')['goals'].sum().reset_index()
                    fig = go.Figure(data=[go.Bar(
                        x=interval_data['time_interval'], 
                        y=interval_data['goals'],
                        marker_color='red')])
                    fig.update_layout(
                            title='Distribution of Goals by Time Intervals',
                            xaxis_title='Time Intervals (minutes)',
                            yaxis_title='Number of Goals',
                            template='plotly_white')
                    st.plotly_chart(fig)


    
    if player_name != 'All Players':
            data = load_data()
            col1, col2 = st.columns(2)
            with col1:
                data = data[data['player']==player_name]
                mins = data.groupby('minute').size().reset_index()
                mins.columns = ['minute','shots']
                goals = data[data['goal']==True].reset_index()
                goals = goals.groupby('minute').size().reset_index()
                goals.columns = ['minute','goals']
                fig = create_pitch_plot_player(data)
                fig.update_layout(title='Shot distribution')
                #final_fig.update_layout(width=4000)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                bins = [0, 15, 30, 45, 60, 75, 90]
                labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                mins['time_interval'] = pd.cut(mins['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                interval_data = mins.groupby('time_interval')['shots'].sum().reset_index()
                fig = go.Figure(data=[go.Bar(
                    x=interval_data['time_interval'], 
                    y=interval_data['shots'],
                    marker_color='blue')])
                fig.update_layout(
                        title='Distribution of Shots by Time Intervals',
                        xaxis_title='Time Intervals (minutes)',
                        yaxis_title='Number of Shots',
                        template='plotly_white')
                st.plotly_chart(fig)
                bins = [0, 15, 30, 45, 60, 75, 90]
                labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
                goals['time_interval'] = pd.cut(goals['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
                interval_data = goals.groupby('time_interval')['goals'].sum().reset_index()
                fig = go.Figure(data=[go.Bar(
                    x=interval_data['time_interval'], 
                    y=interval_data['goals'],
                    marker_color='red')])
                fig.update_layout(
                        title='Distribution of Goals by Time Intervals',
                        xaxis_title='Time Intervals (minutes)',
                        yaxis_title='Number of Goals',
                        template='plotly_white')
                st.plotly_chart(fig)
    
if fora != 'For':
    shotstype = st.selectbox('Select shot type: ',options = ['All shots','From Set Pieces'])
    data = load_opp_data()
    if shotstype == 'All shots':
        col1, col2 = st.columns(2)
        with col1:
            data = data
            mins = data.groupby('minute').size().reset_index()
            mins.columns = ['minute','shots']
            goals = data[data['goal']==True].reset_index()
            goals = goals.groupby('minute').size().reset_index()
            goals.columns = ['minute','goals']
            fig = create_pitch_plot_all(data)
            fig.update_layout(title='Shot distribution')
            #final_fig.update_layout(width=4000)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            bins = [0, 15, 30, 45, 60, 75, 90]
            labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
            mins['time_interval'] = pd.cut(mins['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
            interval_data = mins.groupby('time_interval')['shots'].sum().reset_index()
            fig = go.Figure(data=[go.Bar(
                x=interval_data['time_interval'], 
                y=interval_data['shots'],
                marker_color='blue')])
            fig.update_layout(
                    title='Distribution of Shots by Time Intervals',
                    xaxis_title='Time Intervals (minutes)',
                    yaxis_title='Number of Shots',
                    template='plotly_white')
            st.plotly_chart(fig)
            bins = [0, 15, 30, 45, 60, 75, 90]
            labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
            goals['time_interval'] = pd.cut(goals['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
            interval_data = goals.groupby('time_interval')['goals'].sum().reset_index()
            fig = go.Figure(data=[go.Bar(
                x=interval_data['time_interval'], 
                y=interval_data['goals'],
                marker_color='red')])
            fig.update_layout(
                    title='Distribution of Goals by Time Intervals',
                    xaxis_title='Time Intervals (minutes)',
                    yaxis_title='Number of Goals',
                    template='plotly_white')
            st.plotly_chart(fig)
    if shotstype != 'All shots':
        col1, col2 = st.columns(2)
        with col1:
            data = data[data['possession'].str.contains('Set piece|corner')].reset_index(drop='index')
            mins = data.groupby('minute').size().reset_index()
            mins.columns = ['minute','shots']
            goals = data[data['goal']==True].reset_index()
            goals = goals.groupby('minute').size().reset_index()
            goals.columns = ['minute','goals']
            fig = create_pitch_plot_all(data)
            fig.update_layout(title='Shot distribution')
            #final_fig.update_layout(width=4000)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            bins = [0, 15, 30, 45, 60, 75, 90]
            labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
            mins['time_interval'] = pd.cut(mins['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
            interval_data = mins.groupby('time_interval')['shots'].sum().reset_index()
            fig = go.Figure(data=[go.Bar(
                x=interval_data['time_interval'], 
                y=interval_data['shots'],
                marker_color='blue')])
            fig.update_layout(
                    title='Distribution of Shots by Time Intervals',
                    xaxis_title='Time Intervals (minutes)',
                    yaxis_title='Number of Shots',
                    template='plotly_white')
            st.plotly_chart(fig)
            bins = [0, 15, 30, 45, 60, 75, 90]
            labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]
            goals['time_interval'] = pd.cut(goals['minute'], bins=bins, labels=labels, right=False, include_lowest=True)
            interval_data = goals.groupby('time_interval')['goals'].sum().reset_index()
            fig = go.Figure(data=[go.Bar(
                x=interval_data['time_interval'], 
                y=interval_data['goals'],
                marker_color='red')])
            fig.update_layout(
                    title='Distribution of Goals by Time Intervals',
                    xaxis_title='Time Intervals (minutes)',
                    yaxis_title='Number of Goals',
                    template='plotly_white')
            st.plotly_chart(fig)
