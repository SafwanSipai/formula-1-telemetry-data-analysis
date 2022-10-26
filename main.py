import math
import matplotlib
import streamlit as st
import fastf1
import pandas as pd
import datetime
import fastf1.plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np

fastf1.Cache.enable_cache('cache') 
plt.style.use('dark_background')

def racetime_to_format(x):
    if x['Time'] != '-':
        total = x['Time'].total_seconds()
        m = int(total/60)
        h = int(m/60)
        m = int(m%60)
        s = total%60
        ms = (s-int(s))
        ms = round(ms, 3)
        ms = str(ms)[2:]
        if len(ms)<3:
            ms += '0'
        if h>0:
            t = f'{h}:{m}:{int(s)}:{ms}'
        else:
            t = f'{m}:{int(s)}:{ms}'         
    else:
        if x['Status'][0] == '+':
            t = x['Status']
        else:
            t = 'DNF'
    
    return t

def qualitime_to_format(x):
    if x != '-':
        total = x.total_seconds()
        m = int(total/60)
        h = int(m/60)
        s = total%60
        ms = (s-int(s))
        ms = round(ms, 3)
        ms = str(ms)[2:]
        if len(ms)<3:
            ms += '0'
        if h>0:
            t = f'{h}:{m}:{int(s)}:{ms}'
        t = f'{m}:{int(s)}:{ms}'        
    else:
        t = '-'
        
    return t

def load_session(year, track):
    race = fastf1.get_session(year, track, 'R')
    qualifying = fastf1.get_session(year, track, 'Q')
    return race, qualifying

def race_time(race_session):
    race_session['Time'].replace({pd.NaT:'-'}, inplace=True)

    raceTime = pd.DataFrame(columns=['Time'])
    raceTime['Time'] = race_session.apply(lambda x:racetime_to_format(x), axis=1)
    return raceTime


def show_result_race(session):
    result = session.results[['Position', 'FullName']]
    result.rename(columns={'FullName':'Full Name'}, inplace=True)
    result = pd.DataFrame(result)
    raceTime = race_time(session.results)

    for pos in range(len(result['Position'])):
        if(not math.isnan(result['Position'][pos])):
            result['Position'][pos] = int(result['Position'][pos])
        else:
            result['Position'][pos] = int(result['Position'][pos-1]+1)

    result['Position'] = result['Position'].apply(lambda x:int(x))
    result = pd.concat([result, raceTime], axis=1)
    st.write(result)

def quali_sessions(result):

    result['Q1'].replace({pd.NaT:'-'}, inplace=True)
    result['Q2'].replace({pd.NaT:'-'}, inplace=True)
    result['Q3'].replace({pd.NaT:'-'}, inplace=True)

    # st.write(result)
    # new_result = result[['Q1', 'Q2', 'Q3']]

    # st.write(new_result)
    result['Q1'] = result['Q1'].apply(lambda x:qualitime_to_format(x))
    result['Q2'] = result['Q2'].apply(lambda x:qualitime_to_format(x))
    result['Q3'] = result['Q3'].apply(lambda x:qualitime_to_format(x))

    return result['Q1'], result['Q2'], result['Q3']


def show_result_qualifying(session):
    result = session.results[['Position', 'FullName']]
    result.rename(columns={'FullName':'Full Name'}, inplace=True)
    result = pd.DataFrame(result)
    q1, q2, q3 = quali_sessions(session.results)

    for pos in range(len(result['Position'])):
        if(not math.isnan(result['Position'][pos])):
            result['Position'][pos] = int(result['Position'][pos])
        else:
            result['Position'][pos] = int(result['Position'][pos-1]+1)
    
    result['Position'] = result['Position'].apply(lambda x:int(x))
    result = pd.concat([result, q1, q2, q3], axis=1)
    st.write(result)

def lap_time_number(laps):
    lapTime = []
    lapNumber = []
    for lap in laps:
        lapNumber.append(lap[0])
        lapTime.append(lap[1][2])

    first = lapNumber[0]
    for i in range(len(lapNumber)):
        lapNumber[i] = lapNumber[i]-first

    lapTime = lapTime[1:]
    lapNumber = lapNumber[1:]

    for i in range(len(lapTime)):
        lapTime[i] = lapTime[i].total_seconds()
        lapTime[i] = lapTime[i]/60

    lapTime = pd.to_datetime(lapTime, unit='m')

    return lapTime, lapNumber

st.image('f1logo.png')
st.title('Formula 1 Data Analysis')

st.write('''
The easy and convenient way to compare the huge amount of data that the fast-moving Forumla 1 cars of 
today generate!''')

seasons = [2018, 2019, 2020, 2021]

year2018 = ['Australia', 'Bahrain', 'China', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'France', 'Austria', 
'Great Britain', 'Hockenheimring', 'Hungary', 'Belgium', 'Monza', 'Singapore', 'Russia', 'Japan', 'Austin',
'Mexico', 'Brazil', 'Abu Dhabi']

year2019 = ['Australia', 'Bahrain', 'China', 'Azerbaijan', 'Spain', 'Monaco', 'Canada', 'France', 'Austria', 
'Great Britain', 'Hockenheimring', 'Hungary', 'Belgium', 'Monza', 'Singapore', 'Russia', 'Japan', 'Austin',
'Mexico', 'Brazil', 'Abu Dhabi']

year2020 = ['Austria', 'Styria', 'Hungary', 'Great Britain', '70th Anniversary', 'Spain', 'Spa', 'Monza', 
'Tuscany', 'Russia', 'Nurburgring', 'Portimao', 'Imola', 'Turkey', 'Bahrain', 'Sakhir', 'Abu Dhabi']

year2021 = ['Bahrain', 'Imola', 'Portimao', 'Spain', 'Monaco', 'Azerbaijan', 'France', 'Styria', 'Austria', 
'Great Britain', 'Hungary', 'Belgium', 'Dutch', 'Monza', 'Russia', 'Turkey', 'Austin', 'Mexico', 'Brazil',
'Qatar', 'Saudi Arabia', 'Abu Dhabi']


tab1, tab2, tab3 = st.tabs(['Grand Prix Stats', 'Driver Stats', 'Compare Two Drivers'])

with tab1:
    year = st.selectbox('Year', seasons, key='raceStatsYear')

    if (year == 2018):
        track = st.selectbox('Track', year2018, key=1)
    elif (year == 2019):
            track = st.selectbox('Track', year2019, key=2)
    elif (year == 2020):
            track = st.selectbox('Track', year2020, key=3)
    elif (year == 2021):
            track = st.selectbox('Track', year2021, key=4)


    if(st.button('Search', key=9)):
        race, qualifying = load_session(year, track)
        race.load()
        qualifying.load()
        tabs = st.tabs(['Race', 'Qualifying'])

        with tabs[0]:
            show_result_race(race)
        with tabs[1]:
            show_result_qualifying(qualifying)


with tab2:
    year = st.selectbox('Year', seasons, key='driverStatsYear')

    if (year == 2018):
        track = st.selectbox('Track', year2018, key=5)
    elif (year == 2019):
            track = st.selectbox('Track', year2019, key=6)
    elif (year == 2020):
            track = st.selectbox('Track', year2020, key=7)
    elif (year == 2021):
            track = st.selectbox('Track', year2021, key=8)

    r_tab, q_tab = st.tabs(['Race', 'Qualifying'])   
    with r_tab: 
        race = fastf1.get_session(year, track, 'R')
        race.load()
        drivers = list(race.results.FullName)
        abbreviations = list(race.results.Abbreviation)
        teamColor = list(race.results.TeamColor)
        driver = st.selectbox('Driver', drivers, key=10)
        if st.button('Search'):

            laps = race.laps.pick_driver(abbreviations[drivers.index(driver)]).iterlaps()
            laps = list(laps)

            lapTime, lapNumber = lap_time_number(laps)
            # Lap time during the race
            st.subheader('Lap times during the race')
            fig, ax = plt.subplots()
            ax.plot(lapNumber, lapTime, color='#'+teamColor[drivers.index(driver)])
            ax.set_xlabel('Lap Number')
            ax.set_ylabel('Lap Time [hh:mm:ss]')
            ax.legend()
            plt.grid()
            st.pyplot(fig)
            st.markdown('''---''')

            # Pit stops
            pit = 0
            for lap in laps:
                if(not pd.isnull(lap[1][5])):
                    pit += 1

            st.subheader('Pit Stops')
            st.subheader(pit)
            st.markdown('''---''')

    with q_tab: 
        qualifying = fastf1.get_session(year, track, 'Q')
        qualifying.load()
        drivers = list(qualifying.results.FullName)
        abbreviations = list(qualifying.results.Abbreviation)
        teamColor = list(qualifying.results.TeamColor)
        driver = st.selectbox('Driver', drivers, key=11)
        if st.button('Search', key=12):

            lap = qualifying.laps.pick_driver(abbreviations[drivers.index(driver)]).pick_fastest()
            data = lap.get_car_data().add_distance()
            t = data['Distance']
            vCar = data['Speed']

            # Speeds in the fastest lap
            st.subheader('Speeds for the fastest lap of the session')
            fig, ax = plt.subplots()
            ax.plot(t, vCar, color='#'+teamColor[drivers.index(driver)])
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Speed [Km/h]')
            ax.legend()
            plt.grid()
            st.pyplot(fig)
            st.markdown('''---''')

            throttle = data['Throttle']
            st.subheader('Throttle application in the fastest lap of the session')
            fig, ax = plt.subplots()
            ax.plot(t, throttle, color='#'+teamColor[drivers.index(driver)])
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Throttle [%]')
            ax.legend()
            plt.grid()
            st.pyplot(fig)
            st.markdown('''---''')

            # Gear shifts in the fastest lap
            st.subheader('Gear Shifts in the fastest lap of the session')
            tel = lap.get_telemetry()
            x = np.array(tel['X'].values)
            y = np.array(tel['Y'].values)

            fig = plt.figure()
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            gear = tel['nGear'].to_numpy().astype(float)

            cmap = cm.get_cmap('Paired')
            lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
            lc_comp.set_array(gear)
            lc_comp.set_linewidth(4)

            ax = plt.gca().add_collection(lc_comp)
            plt.axis('equal')
            plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

            cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
            cbar.set_ticks(np.arange(1.5, 9.5))
            cbar.set_ticklabels(np.arange(1, 9))

            st.pyplot(fig)
            st.markdown('''---''')

            # Speed visualisation on track
            st.subheader('Speeds around the fastest lap of the session')
            colormap = matplotlib.cm.plasma
            x = lap.telemetry['X']              
            y = lap.telemetry['Y']              
            color = lap.telemetry['Speed']

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
            ax.axis('off')
            ax.plot(lap.telemetry['X'], lap.telemetry['Y'], color='black', linestyle='-', linewidth=16, zorder=0)
            norm = plt.Normalize(color.min(), color.max())
            lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=5)

            lc.set_array(color)
            line = ax.add_collection(lc)
            cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
            normlegend = matplotlib.colors.Normalize(vmin=color.min(), vmax=color.max())
            legend = matplotlib.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal")

            st.pyplot(fig)
            st.markdown('''---''')


with tab3:

    year = st.selectbox('Year', seasons, key='compareStats')

    if (year == 2018):
        track = st.selectbox('Track', year2018, key=13)
    elif (year == 2019):
            track = st.selectbox('Track', year2019, key=14)
    elif (year == 2020):
            track = st.selectbox('Track', year2020, key=15)
    elif (year == 2021):
            track = st.selectbox('Track', year2021, key=16)

    t1, t2 = st.tabs(['Race', 'Qualifying'])

    with t1:
        race = fastf1.get_session(year, track, 'R')
        race.load()
        drivers = list(race.results.FullName)
        abbreviations = list(race.results.Abbreviation)
        teamColor = list(race.results.TeamColor)
        col7, col8 = st.columns(2)

        with col7:
            
            driver1 = st.selectbox('Driver', drivers, key=19)

        with col8:

            driver2 = st.selectbox('Driver', drivers, key=20)

        if (st.button('Compare', key=21)):
            laps1 = race.laps.pick_driver(abbreviations[drivers.index(driver1)]).iterlaps()
            laps2 = race.laps.pick_driver(abbreviations[drivers.index(driver2)]).iterlaps()

            laps1 = list(laps1)
            laps2 = list(laps2)

            lapTime1, lapNumber1 = lap_time_number(laps1) 
            lapTime2, lapNumber2 = lap_time_number(laps2)

            st.subheader('Lap times during the race')
            fig, ax = plt.subplots()
            ax.plot(lapNumber1, lapTime1, label=abbreviations[drivers.index(driver1)],
            color='#'+teamColor[drivers.index(driver1)])
            ax.plot(lapNumber2, lapTime2, label=abbreviations[drivers.index(driver2)], 
            color='#'+teamColor[drivers.index(driver2)])
            ax.set_xlabel('Lap Number')
            ax.set_ylabel('Lap Time')
            ax.legend()
            plt.grid()
            st.pyplot(fig)
            st.markdown('''---''')

            # Pit stops
            pit1 = 0
            pit2 = 0
            for lap1 in laps1:
                if(not pd.isnull(lap1[1][5])):
                    pit1 += 1

            for lap2 in laps2:
                if(not pd.isnull(lap2[1][5])):
                    pit2 += 1

            st.subheader('Pit Stops')
            col9, col10 = st.columns(2)
            with col9:
                st.write(f'''{driver1}''')
                st.subheader(pit1)
            
            with col10:
                st.write(f'''{driver2}''')
                st.subheader(pit2)
            st.markdown('''---''')

    with t2:

        qualifying = fastf1.get_session(year, track, 'Q')
        qualifying.load()
        drivers = list(qualifying.results.FullName)
        abbreviations = list(qualifying.results.Abbreviation)
        teamColor = list(qualifying.results.TeamColor)
        col1, col2 = st.columns(2)
        d1 = d2 = ddata1 = ddata2 = None

        with col1:
            
            driver1 = st.selectbox('Driver', drivers, key=17)

        with col2:

            driver2 = st.selectbox('Driver', drivers, key=18)

        if (st.button('Compare')):
            d1 = qualifying.laps.pick_driver(abbreviations[drivers.index(driver1)]).pick_fastest()
            ddata1 = d1.get_car_data().add_distance()
            dist1 = ddata1['Distance']
            vCar1 = ddata1['Speed']

            d2 = qualifying.laps.pick_driver(abbreviations[drivers.index(driver2)]).pick_fastest()
            ddata2 = d2.get_car_data().add_distance()
            dist2 = ddata2['Distance']
            vCar2 = ddata2['Speed']

            st.subheader('Speeds for the fastest lap of the session')
            fig, ax = plt.subplots()
            ax.plot(dist1, vCar1, color='#'+teamColor[drivers.index(driver1)],
            label=abbreviations[drivers.index(driver1)])
            ax.plot(dist2, vCar2, color='#'+teamColor[drivers.index(driver2)],
            label=abbreviations[drivers.index(driver2)])
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Speed [Km/h]')
            ax.legend()
            plt.grid()
            st.pyplot(fig)
            st.markdown('''---''')

            throttle1 = ddata1['Throttle']
            throttle2 = ddata2['Throttle']

            st.subheader('Throttle application in the fastest lap of the session')

            fig, ax = plt.subplots()
            ax.plot(dist1, throttle1, color='#'+teamColor[drivers.index(driver1)],
            label=abbreviations[drivers.index(driver1)])
            ax.plot(dist2, throttle2, color='#'+teamColor[drivers.index(driver2)],
            label=abbreviations[drivers.index(driver2)])
            ax.set_xlabel('Distance [m]')
            ax.set_ylabel('Throttle [%]')
            ax.legend()
            plt.grid()
            st.pyplot(fig)
            st.markdown('''---''')

            st.subheader('Gear Shifts in the fastest lap of the session')

            col3, col4 = st.columns(2)
            with col3:

                tel = d1.get_telemetry()
                x = np.array(tel['X'].values)
                y = np.array(tel['Y'].values)

                fig = plt.figure()
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                gear = tel['nGear'].to_numpy().astype(float)

                cmap = cm.get_cmap('Paired')
                lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
                lc_comp.set_array(gear)
                lc_comp.set_linewidth(4)

                ax = plt.gca().add_collection(lc_comp)
                plt.axis('equal')
                plt.suptitle(driver1)
                plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

                cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
                cbar.set_ticks(np.arange(1.5, 9.5))
                cbar.set_ticklabels(np.arange(1, 9))

                col3.pyplot(fig)


            with col4:

                tel = d2.get_telemetry()
                x = np.array(tel['X'].values)
                y = np.array(tel['Y'].values)

                fig = plt.figure()
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                gear = tel['nGear'].to_numpy().astype(float)

                cmap = cm.get_cmap('Paired')
                lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
                lc_comp.set_array(gear)
                lc_comp.set_linewidth(4)

                ax = plt.gca().add_collection(lc_comp)
                plt.axis('equal')
                plt.suptitle(driver2)
                plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

                cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
                cbar.set_ticks(np.arange(1.5, 9.5))
                cbar.set_ticklabels(np.arange(1, 9))

                col4.pyplot(fig)  

            st.markdown('''---''')
            st.subheader('Speeds around the fastest lap of the session')  

            col5, col6 = st.columns(2)

            with col5:
                colormap = matplotlib.cm.plasma
                x = d1.telemetry['X']              
                y = d1.telemetry['Y']              
                color = d1.telemetry['Speed']

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                fig, ax = plt.subplots()
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
                ax.axis('off')
                ax.plot(d1.telemetry['X'], d1.telemetry['Y'], color='black', linestyle='-', linewidth=16, zorder=0)
                norm = plt.Normalize(color.min(), color.max())
                lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=5)

                lc.set_array(color)
                line = ax.add_collection(lc)
                cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
                normlegend = matplotlib.colors.Normalize(vmin=color.min(), vmax=color.max())
                legend = matplotlib.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal")

                plt.suptitle(driver1)
                st.pyplot(fig)

            with col6:
                colormap = matplotlib.cm.plasma
                x = d2.telemetry['X']              
                y = d2.telemetry['Y']              
                color = d2.telemetry['Speed']

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                fig, ax = plt.subplots()
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
                ax.axis('off')
                ax.plot(d2.telemetry['X'], d2.telemetry['Y'], color='black', linestyle='-', linewidth=16, zorder=0)
                norm = plt.Normalize(color.min(), color.max())
                lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=5)

                lc.set_array(color)
                line = ax.add_collection(lc)
                cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.05])
                normlegend = matplotlib.colors.Normalize(vmin=color.min(), vmax=color.max())
                legend = matplotlib.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal")

                plt.suptitle(driver2)
                st.pyplot(fig)

            st.markdown('''---''')
        
