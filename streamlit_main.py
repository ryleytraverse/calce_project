import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

def main():
    subcycle_tab, cycling_tab = st.tabs(['Subcycle Data', 'Cycling Data'])

    with subcycle_tab:
        title_container = st.conatiner()
        plot_container_go = st.container()
        plot_container_plt = st.conatiner()
        
        with title_container:
            st.title('Welcome to my dashboard!')
        
        with plot_container_go:
            fig_go = go.Figure()
            fig_go.add_trace(go.Scatter(x=idk, y=idk2, mode='markers', name='PLOT', line=dict(color='blue')))
            t = 'This is a plot'
            fig_go.update_layout(title=t, xaxis_title='X', yaxis_title='Y')
            st.plotly_chart(fig_go)

        with plot_container_plt:
            fig_plt, ax = plt.subplots()
            ax.scatter(idk, idk2, s=10, color='cyan', label='Data')
            plt.title('This is a plot')
            plt.xlabel('X')
            plt.ylabel('Y')
            leg = plt.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
            st.pyplot(fig_plt)

if __name__ == '__main__':
    main()
